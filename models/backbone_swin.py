from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _resolve_gn_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, h // window_size, w // window_size


def _window_reverse(windows: torch.Tensor, window_size: int, h_blocks: int, w_blocks: int, batch: int) -> torch.Tensor:
    x = windows.view(batch, h_blocks, w_blocks, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, h_blocks * window_size, w_blocks * window_size, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class StaticCrossAttention(nn.Module):
    """Cross-attention: Q=dynamic, K=static, V=dynamic. Static can't paint."""

    def __init__(self, dim: int, static_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(static_dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.weight.data.mul_(0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, static_ctx: torch.Tensor) -> torch.Tensor:
        b_, n, c = x.shape
        nh = self.num_heads
        hd = c // nh
        q = self.q_proj(x).reshape(b_, n, nh, hd).permute(0, 2, 1, 3)
        k = self.k_proj(static_ctx).reshape(b_, n, nh, hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b_, n, nh, hd).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.out_proj(out)


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        static_cross_attn: bool = False,
        static_dim: int = 32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.cross_attn: StaticCrossAttention | None = None
        self.norm_cross: nn.LayerNorm | None = None
        if static_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = StaticCrossAttention(dim, static_dim, num_heads)

    def _build_attn_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor | None:
        if self.shift_size <= 0:
            return None
        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[:, hs, ws_, :] = cnt
                cnt += 1
        mask_windows, hb, wb = _window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        static_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[-2], x.shape[-1]

        if film is not None:
            gamma, beta = film
            if gamma.dim() == 2:
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
            if gamma.shape[-2:] != x.shape[-2:]:
                gamma = F.interpolate(gamma, size=x.shape[-2:], mode="bilinear", align_corners=False)
                beta = F.interpolate(beta, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = gamma * x + beta

        shortcut = x
        x_nhwc = x.permute(0, 2, 3, 1)

        if self.shift_size > 0:
            shifted = torch.roll(x_nhwc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x_nhwc

        attn_mask = self._build_attn_mask(hp, wp, x.device)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(b, 1, 1)
        x_windows, hb, wb = _window_partition(shifted, ws)
        x_windows = x_windows.view(-1, ws * ws, c)
        attn_out = self.attn(self.norm1(x_windows), attn_mask)
        if self.cross_attn is not None and static_ctx is not None:
            sc = static_ctx
            if sc.shape[-2:] != (hp, wp):
                sc = F.interpolate(sc, size=(hp, wp), mode="bilinear", align_corners=False)
            sc_nhwc = sc.permute(0, 2, 3, 1)
            if self.shift_size > 0:
                sc_nhwc = torch.roll(sc_nhwc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            sc_win, _, _ = _window_partition(sc_nhwc, ws)
            sc_flat = sc_win.view(-1, ws * ws, sc.shape[1])
            attn_out = attn_out + self.cross_attn(self.norm_cross(attn_out), sc_flat)
        attn_out = attn_out.view(-1, ws, ws, c)
        shifted = _window_reverse(attn_out, ws, hb, wb, b)
        if self.shift_size > 0:
            x_nhwc = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_nhwc = shifted
        x_nhwc = shortcut.permute(0, 2, 3, 1) + x_nhwc
        x_nhwc = x_nhwc + self.mlp(self.norm2(x_nhwc))
        x = x_nhwc.permute(0, 3, 1, 2)
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x


class GlobalAttentionBlock(nn.Module):
    """Downsampled global self-attention block (no window partitioning)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        downsample_factor: int = 4,
        static_cross_attn: bool = False,
        static_dim: int = 32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.downsample_factor = downsample_factor
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        self.cross_attn: StaticCrossAttention | None = None
        self.norm_cross: nn.LayerNorm | None = None
        if static_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = StaticCrossAttention(dim, static_dim, num_heads)

    def _attn_fn(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Self-attention on a sequence [B, N, C]."""
        b, n, c = x_seq.shape
        nh = self.num_heads
        hd = c // nh
        qkv = self.qkv(x_seq).reshape(b, n, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)

    def forward(
        self,
        x: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        static_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        # FiLM conditioning
        if film is not None:
            gamma, beta = film
            if gamma.dim() == 2:
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
            if gamma.shape[-2:] != x.shape[-2:]:
                gamma = F.interpolate(gamma, size=x.shape[-2:], mode="bilinear", align_corners=False)
                beta = F.interpolate(beta, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = gamma * x + beta

        # Store shortcut
        shortcut = x

        # Downsample via avg_pool2d
        ds = self.downsample_factor
        h_ds = max(1, h // ds)
        w_ds = max(1, w // ds)
        x_ds = F.adaptive_avg_pool2d(x, (h_ds, w_ds))  # [B, C, h_ds, w_ds]

        # Flatten to sequence
        x_seq = x_ds.permute(0, 2, 3, 1).reshape(b, h_ds * w_ds, c)  # [B, N_ds, C]
        x_seq = self.norm1(x_seq)

        # Self-attention with gradient checkpointing
        attn_out = cp.checkpoint(self._attn_fn, x_seq, use_reentrant=False)

        # Cross-attention if available
        if self.cross_attn is not None and static_ctx is not None:
            sc = static_ctx
            if sc.shape[-2:] != (h_ds, w_ds):
                sc = F.interpolate(sc, size=(h_ds, w_ds), mode="bilinear", align_corners=False)
            sc_seq = sc.permute(0, 2, 3, 1).reshape(b, h_ds * w_ds, sc.shape[1])
            attn_out = attn_out + self.cross_attn(self.norm_cross(attn_out), sc_seq)

        # Reshape back to spatial and upsample
        attn_out = attn_out.reshape(b, h_ds, w_ds, c).permute(0, 3, 1, 2)  # [B, C, h_ds, w_ds]
        attn_out = F.interpolate(attn_out, size=(h, w), mode="bilinear", align_corners=False)

        # Residual add
        x = shortcut + attn_out

        # MLP with residual (on full resolution)
        x_nhwc = x.permute(0, 2, 3, 1)
        x_nhwc = x_nhwc + self.mlp(self.norm2(x_nhwc))
        x = x_nhwc.permute(0, 3, 1, 2)

        return x


class BackboneSwin(nn.Module):
    """L0 full-resolution Swin context encoder (replaces U-Net backbone in v2)."""

    STAGES = ("swin0", "swin1", "swin2")

    def __init__(
        self,
        in_channels: int = 72,
        *,
        window_size: int = 8,
        num_heads: int = 4,
        depth_per_stage: int = 2,
        mlp_ratio: float = 4.0,
        static_cross_attn: bool = False,
        static_cross_dim: int = 32,
        global_attn_downsample: int = 4,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by num_heads={num_heads}")
        self.in_channels = in_channels
        self.window_size = window_size
        self.static_cross_attn_enabled = static_cross_attn
        self.gradient_checkpointing = gradient_checkpointing
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_gn_groups(in_channels), in_channels),
            nn.SiLU(inplace=True),
        )
        self.static_proj: nn.Module | None = None
        if static_cross_attn:
            self.static_proj = nn.Sequential(
                nn.Conv2d(10, static_cross_dim, kernel_size=3, padding=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(static_cross_dim, static_cross_dim, kernel_size=3, padding=1, bias=True),
            )
        blocks: list[nn.Module] = []
        for _stage_idx in range(len(self.STAGES)):
            for block_idx in range(depth_per_stage):
                shift = window_size // 2 if block_idx % 2 == 1 else 0
                is_stage_last = (block_idx == depth_per_stage - 1)
                if is_stage_last:
                    blocks.append(
                        GlobalAttentionBlock(
                            dim=in_channels,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            downsample_factor=global_attn_downsample,
                            static_cross_attn=static_cross_attn,
                            static_dim=static_cross_dim,
                        )
                    )
                else:
                    blocks.append(
                        SwinBlock(
                            dim=in_channels,
                            window_size=window_size,
                            num_heads=num_heads,
                            shift_size=shift,
                            mlp_ratio=mlp_ratio,
                            static_cross_attn=False,
                            static_dim=static_cross_dim,
                        )
                    )
        self.blocks = nn.ModuleList(blocks)
        self.stage_boundaries = [depth_per_stage * (i + 1) for i in range(len(self.STAGES))]

    def forward(
        self,
        x: torch.Tensor,
        film_params: Mapping[str, Tuple[torch.Tensor, torch.Tensor]] | None = None,
        branch_inject: torch.Tensor | None = None,
        branch_gate: torch.Tensor | None = None,
        inject_after_stage: int = 1,
        static_16: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.stem(x)
        static_ctx = None
        if self.static_proj is not None and static_16 is not None:
            from static_spatial_film import slice_static_active
            sf = slice_static_active(static_16)
            if sf.shape[-2:] != x.shape[-2:]:
                sf = F.interpolate(sf, size=x.shape[-2:], mode="bilinear", align_corners=False)
            static_ctx = self.static_proj(sf)

        stage_idx = 0
        for block_idx, block in enumerate(self.blocks):
            stage_key = self.STAGES[min(stage_idx, len(self.STAGES) - 1)]
            film = film_params.get(stage_key) if film_params else None
            if self.gradient_checkpointing and self.training:
                x = cp.checkpoint(block, x, film, static_ctx, use_reentrant=False)
            else:
                x = block(x, film=film, static_ctx=static_ctx)
            if block_idx + 1 == self.stage_boundaries[inject_after_stage]:
                if branch_inject is not None and branch_gate is not None:
                    bmap = branch_inject
                    if bmap.shape[-2:] != x.shape[-2:]:
                        bmap = F.interpolate(bmap, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    gate = branch_gate
                    if gate.shape[1] != bmap.shape[1]:
                        gate = gate.mean(dim=1, keepdim=True)
                    x = x + gate * bmap
            if block_idx + 1 in self.stage_boundaries:
                stage_idx += 1
        return x


__all__ = ["BackboneSwin", "GlobalAttentionBlock", "StaticCrossAttention"]

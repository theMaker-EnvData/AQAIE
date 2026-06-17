from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _resolve_gn_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


def _apply_film_2d(x: torch.Tensor, film: nn.Linear, time_context: torch.Tensor) -> torch.Tensor:
    if time_context.dim() != 2 or time_context.shape[0] != x.shape[0]:
        raise ValueError(
            f"time_context must be [B,D] with matching batch, got {tuple(time_context.shape)} for B={x.shape[0]}"
        )
    channels = x.shape[1]
    raw = film(time_context)
    d_gamma, beta = torch.split(raw, [channels, channels], dim=1)
    gamma = 1.0 + d_gamma
    return x * gamma[:, :, None, None] + beta[:, :, None, None]


def resolve_inr_encoder_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        config = {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    obs_cfg = model_cfg.get("obs_encoder", {}) if isinstance(model_cfg.get("obs_encoder"), Mapping) else {}
    inr_cfg = model_cfg.get("inr", {}) if isinstance(model_cfg.get("inr"), Mapping) else {}
    mlp_hidden_raw = inr_cfg.get("mlp_hidden", obs_cfg.get("mlp_hidden", [128, 128]))
    if isinstance(mlp_hidden_raw, (list, tuple)) and len(mlp_hidden_raw) >= 2:
        mlp_hidden = (int(mlp_hidden_raw[0]), int(mlp_hidden_raw[1]))
    else:
        mlp_hidden = (128, 128)
    dist_cfg = inr_cfg.get("distance_bias", {}) if isinstance(inr_cfg.get("distance_bias"), Mapping) else {}
    jitter_raw = inr_cfg.get("pe_jitter", False)
    if isinstance(jitter_raw, Mapping):
        pe_jitter_enabled = bool(jitter_raw.get("enabled", False))
    else:
        pe_jitter_enabled = bool(jitter_raw)
    return {
        "out_channels": int(inr_cfg.get("out_channels", 24)),
        "time_steps": int(inr_cfg.get("time_steps", 3)),
        "time_context_dim": int(inr_cfg.get("time_context_dim", 512)),
        "met_ctx_channels": int(inr_cfg.get("met_ctx_channels", 31)),
        "static_ctx_channels": int(inr_cfg.get("static_ctx_channels", 10)),
        "static_ctx_enabled": bool(inr_cfg.get("static_ctx_enabled", True)),
        "embed_dim": int(inr_cfg.get("embed_dim", obs_cfg.get("embed_dim", 128))),
        "num_heads": int(inr_cfg.get("num_heads", obs_cfg.get("num_heads", 4))),
        "cross_attn_layers": int(inr_cfg.get("cross_attn_layers", obs_cfg.get("cross_attn_layers", 2))),
        "mlp_hidden": mlp_hidden,
        "pe_freqs": int(inr_cfg.get("pe_freqs", 16)),
        "pe_mode": str(inr_cfg.get("pe_mode", "normalized")).strip().lower(),
        "pe_base_wavelength_cells": float(inr_cfg.get("pe_base_wavelength_cells", 2.0)),
        "pe_jitter_enabled": pe_jitter_enabled,
        "num_pollutants": int(inr_cfg.get("num_pollutants", 6)),
        "field_heads": bool(inr_cfg.get("field_heads", True)),
        "temporal_layers": int(inr_cfg.get("temporal_layers", 1)),
        "distance_bias_enabled": bool(dist_cfg.get("enabled", False)),
        "distance_bias_init_sigma_cells": float(dist_cfg.get("init_sigma_cells", 12.0)),
        "distance_bias_min_sigma_cells": float(dist_cfg.get("min_sigma_cells", 1.0)),
        "attn_query_chunk": int(inr_cfg.get("attn_query_chunk", 0)),
        "station_topk": int(inr_cfg.get("station_topk", 0)),
    }


class FourierPosEncoding2d(nn.Module):
    """Fixed 2D Fourier features for normalized coordinates in [-1, 1].

    Legacy mode: same coordinate value means different physical distance on
    different grid sizes. Use FourierPosEncodingCells for crop/size-invariant
    encodings.
    """

    def __init__(self, num_freqs: int = 16) -> None:
        super().__init__()
        freqs = torch.pow(2.0, torch.arange(num_freqs, dtype=torch.float32)) * math.pi
        self.register_buffer("_freqs", freqs, persistent=False)
        self.out_dim = 4 * num_freqs

    def forward(self, coord_norm: torch.Tensor) -> torch.Tensor:
        x = coord_norm[..., 0:1] * self._freqs
        y = coord_norm[..., 1:2] * self._freqs
        return torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)


class FourierPosEncodingCells(nn.Module):
    """Fixed 2D Fourier features of cell-unit coordinates with fixed wavelengths.

    Wavelengths are base * 2^i cells (i.e. physical km on a 1 km grid), so the
    encoding is invariant to grid size and crop position-relative geometry,
    unlike grid-normalized coordinates.
    """

    def __init__(self, num_freqs: int = 8, base_wavelength_cells: float = 2.0) -> None:
        super().__init__()
        wavelengths = float(base_wavelength_cells) * torch.pow(2.0, torch.arange(num_freqs, dtype=torch.float32))
        freqs = 2.0 * math.pi / wavelengths
        self.register_buffer("_freqs", freqs, persistent=False)
        self.out_dim = 4 * num_freqs
        # Largest wavelength: uniform origin jitter over this range gives a
        # uniform phase for every (power-of-two) frequency in the bank.
        self.max_wavelength_cells = float(wavelengths[-1].item()) if num_freqs > 0 else 0.0

    def forward(self, coord_cells: torch.Tensor) -> torch.Tensor:
        x = coord_cells[..., 0:1] * self._freqs
        y = coord_cells[..., 1:2] * self._freqs
        return torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)


class CrossAttnBlock(nn.Module):
    """Cross-attention block on SDPA (flash/mem-efficient kernels).

    Optional Gaussian distance bias injects spatial locality into attention
    logits: bias = -d^2 / (2*sigma_h^2) with a learnable per-head sigma
    (grid-cell units). Query chunking bounds the [B,h,Lq,N] bias memory.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        *,
        distance_bias: bool = False,
        init_sigma_cells: float = 12.0,
        min_sigma_cells: float = 1.0,
        query_chunk: int = 0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.query_chunk = max(0, int(query_chunk))
        self.min_sigma_cells = float(min_sigma_cells)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if distance_bias:
            init_sigma = max(float(init_sigma_cells), self.min_sigma_cells)
            self.log_sigma = nn.Parameter(torch.full((self.num_heads,), math.log(init_sigma)))
        else:
            self.log_sigma = None

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim),
        )

    def _attn_bias(
        self,
        key_padding_mask: torch.Tensor | None,
        dist2: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Combined additive bias [B, h, Lq, N] (or broadcastable slices thereof)."""
        bias = None
        if self.log_sigma is not None and dist2 is not None:
            sigma = self.log_sigma.exp().clamp_min(self.min_sigma_cells)
            inv_two_sigma2 = 0.5 / sigma.square()  # [h]
            bias = -dist2.unsqueeze(1).to(dtype=dtype) * inv_two_sigma2.view(1, -1, 1, 1).to(dtype=dtype)
        if key_padding_mask is not None:
            neg = torch.finfo(dtype).min
            pad = torch.where(
                key_padding_mask[:, None, None, :],
                torch.full((), neg, dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
            )
            bias = pad if bias is None else bias + pad
        return bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        dist2: torch.Tensor | None = None,
        topk_idx: torch.Tensor | None = None,
        dist2_topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query: [B, Lq, D], key/value: [B, N, D]
        # key_padding_mask: [B, N] True=pad, dist2: [B, Lq, N] squared cell distance
        # topk_idx/dist2_topk: [B, Lq, K] per-query nearest-station selection
        b, lq, d = query.shape
        if topk_idx is not None:
            attn = self._attend_topk(query, key, value, key_padding_mask, topk_idx, dist2_topk)
        else:
            attn = self._attend_dense(query, key, value, key_padding_mask, dist2)
        attn_out = self.out_proj(attn.reshape(b, lq, d))

        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

    def _attend_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        dist2: torch.Tensor | None,
    ) -> torch.Tensor:
        b, lq, _ = query.shape
        n = key.shape[1]
        q = self.q_proj(query).view(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        chunk = self.query_chunk if self.query_chunk > 0 else lq
        outs: list[torch.Tensor] = []
        for start in range(0, lq, chunk):
            end = min(lq, start + chunk)
            dist2_chunk = None if dist2 is None else dist2[:, start:end, :]
            bias = self._attn_bias(key_padding_mask, dist2_chunk, dtype=q.dtype, device=q.device)
            out_chunk = F.scaled_dot_product_attention(
                q[:, :, start:end, :],
                k,
                v,
                attn_mask=bias,
            )
            outs.append(out_chunk)
        attn = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return attn.transpose(1, 2)  # [B, Lq, h, hd]

    def _topk_chunk_attn(
        self,
        q_c: torch.Tensor,  # [B, lc, h, hd]
        idx: torch.Tensor,  # [B, lc*K]
        d2_c: torch.Tensor | None,  # [B, lc, K] or None
        k_full: torch.Tensor,  # [B, N, D]
        v_full: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, lc = q_c.shape[0], q_c.shape[1]
        d = k_full.shape[-1]
        kk = idx.shape[-1] // lc
        scale = 1.0 / math.sqrt(self.head_dim)
        k_g = k_full.gather(1, idx.unsqueeze(-1).expand(-1, -1, d)).view(b, lc, kk, self.num_heads, self.head_dim)
        v_g = v_full.gather(1, idx.unsqueeze(-1).expand(-1, -1, d)).view(b, lc, kk, self.num_heads, self.head_dim)
        logits = torch.einsum("blhd,blkhd->blhk", q_c, k_g) * scale  # [B, lc, h, K]
        if self.log_sigma is not None and d2_c is not None:
            sigma = self.log_sigma.exp().clamp_min(self.min_sigma_cells)
            sigma_bias_scale = (0.5 / sigma.square()).view(1, 1, -1, 1)
            logits = logits - d2_c.to(dtype=logits.dtype).unsqueeze(2) * sigma_bias_scale.to(dtype=logits.dtype)
        pad_g = None
        if key_padding_mask is not None:
            pad_g = key_padding_mask.gather(1, idx).view(b, lc, 1, kk)
            logits = logits.masked_fill(pad_g, torch.finfo(logits.dtype).min)
        attn_w = torch.softmax(logits, dim=-1)
        if pad_g is not None:
            # Rows whose selected stations are all padded yield uniform garbage; zero them.
            attn_w = attn_w.masked_fill(pad_g.all(dim=-1, keepdim=True), 0.0)
        return torch.einsum("blhk,blkhd->blhd", attn_w, v_g)

    def _attend_topk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        topk_idx: torch.Tensor,
        dist2_topk: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-query attention over K nearest stations only.

        Bounds compute to O(Lq*K) regardless of station count. Chunks are
        gradient-checkpointed during training: the gathered [lc, K, D] key/value
        activations would otherwise cost O(Lq*K*D) memory per layer per lag,
        which OOMs on large crops / full domain.
        """
        b, lq, d = query.shape
        kk = topk_idx.shape[-1]
        q = self.q_proj(query).view(b, lq, self.num_heads, self.head_dim)
        k_full = self.k_proj(key)  # [B, N, D]
        v_full = self.v_proj(value)

        use_ckpt = torch.is_grad_enabled() and (
            q.requires_grad or k_full.requires_grad or v_full.requires_grad
        )
        chunk = self.query_chunk if self.query_chunk > 0 else lq
        outs: list[torch.Tensor] = []
        for start in range(0, lq, chunk):
            end = min(lq, start + chunk)
            lc = end - start
            idx = topk_idx[:, start:end, :].reshape(b, lc * kk)  # [B, lc*K]
            d2_c = None if dist2_topk is None else dist2_topk[:, start:end, :]
            q_c = q[:, start:end]
            if use_ckpt:
                out_chunk = torch.utils.checkpoint.checkpoint(
                    self._topk_chunk_attn,
                    q_c,
                    idx,
                    d2_c,
                    k_full,
                    v_full,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                out_chunk = self._topk_chunk_attn(q_c, idx, d2_c, k_full, v_full, key_padding_mask)
            outs.append(out_chunk)
        attn = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return attn  # [B, Lq, h, hd]


class ObsSparseINREncoder(nn.Module):
    """Parquet-native sparse obs -> dense feature map via met/static-conditioned implicit field."""

    def __init__(
        self,
        out_channels: int = 24,
        time_steps: int = 3,
        time_context_dim: int = 512,
        met_ctx_channels: int = 31,
        static_ctx_channels: int = 10,
        static_ctx_enabled: bool = True,
        embed_dim: int = 128,
        num_heads: int = 4,
        cross_attn_layers: int = 2,
        mlp_hidden: tuple[int, int] = (128, 128),
        pe_freqs: int = 16,
        pe_mode: str = "normalized",
        pe_base_wavelength_cells: float = 2.0,
        pe_jitter_enabled: bool = False,
        num_pollutants: int = 6,
        field_heads: bool = True,
        temporal_layers: int = 1,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            resolved = resolve_inr_encoder_config(config)
            out_channels = resolved["out_channels"]
            time_steps = resolved["time_steps"]
            time_context_dim = resolved["time_context_dim"]
            met_ctx_channels = resolved["met_ctx_channels"]
            static_ctx_channels = resolved["static_ctx_channels"]
            static_ctx_enabled = resolved["static_ctx_enabled"]
            embed_dim = resolved["embed_dim"]
            num_heads = resolved["num_heads"]
            cross_attn_layers = resolved["cross_attn_layers"]
            mlp_hidden = resolved["mlp_hidden"]
            pe_freqs = resolved["pe_freqs"]
            pe_mode = resolved["pe_mode"]
            pe_base_wavelength_cells = resolved["pe_base_wavelength_cells"]
            pe_jitter_enabled = resolved["pe_jitter_enabled"]
            num_pollutants = resolved["num_pollutants"]
            field_heads = resolved["field_heads"]
            temporal_layers = resolved["temporal_layers"]
            distance_bias_enabled = resolved["distance_bias_enabled"]
            distance_bias_init_sigma = resolved["distance_bias_init_sigma_cells"]
            distance_bias_min_sigma = resolved["distance_bias_min_sigma_cells"]
            attn_query_chunk = resolved["attn_query_chunk"]
            station_topk = resolved["station_topk"]
        else:
            distance_bias_enabled = False
            distance_bias_init_sigma = 12.0
            distance_bias_min_sigma = 1.0
            attn_query_chunk = 0
            station_topk = 0

        self.out_channels = int(out_channels)
        self.time_steps = int(time_steps)
        self.time_context_dim = int(time_context_dim)
        self.num_pollutants = int(num_pollutants)
        self.static_ctx_channels = int(static_ctx_channels)
        self.static_ctx_enabled = bool(static_ctx_enabled)
        self.field_heads = bool(field_heads)

        pe_mode = str(pe_mode).strip().lower()
        if pe_mode not in ("normalized", "cells"):
            raise ValueError(f"model.inr.pe_mode must be 'normalized' or 'cells', got '{pe_mode}'")
        self.pe_mode = pe_mode
        self.station_topk = max(0, int(station_topk))
        if self.pe_mode == "cells":
            self.pos_enc: nn.Module = FourierPosEncodingCells(
                num_freqs=pe_freqs, base_wavelength_cells=pe_base_wavelength_cells
            )
        else:
            self.pos_enc = FourierPosEncoding2d(num_freqs=pe_freqs)
        # PE origin jitter (train only, cells mode): a shared random coordinate
        # offset per sample decorrelates PE phase from crop-edge position, so
        # edge behavior cannot anchor to coordinate 0 and alias at wavelength
        # multiples (the 64-cell-lattice blob artifact).
        self.pe_jitter_enabled = bool(pe_jitter_enabled) and self.pe_mode == "cells"
        station_in = num_pollutants * 2 + self.pos_enc.out_dim
        self.station_embed = nn.Sequential(
            nn.Linear(station_in, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        query_ctx_channels = met_ctx_channels + (static_ctx_channels if static_ctx_enabled else 0)
        query_in = self.pos_enc.out_dim + query_ctx_channels
        self.query_proj = nn.Sequential(
            nn.Linear(query_in, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.distance_bias_enabled = bool(distance_bias_enabled)
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim=embed_dim * 2,
                    distance_bias=self.distance_bias_enabled,
                    init_sigma_cells=distance_bias_init_sigma,
                    min_sigma_cells=distance_bias_min_sigma,
                    query_chunk=attn_query_chunk,
                )
                for _ in range(cross_attn_layers)
            ]
        )
        h1, h2 = mlp_hidden
        decode_in = embed_dim + query_ctx_channels
        field_decode_in = embed_dim + self.pos_enc.out_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(decode_in, h1),
            nn.SiLU(inplace=True),
            nn.Linear(h1, h2),
            nn.SiLU(inplace=True),
            nn.Linear(h2, out_channels),
        )
        if field_heads:
            self.field_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(field_decode_in, h1),
                        nn.SiLU(inplace=True),
                        nn.Linear(h1, h2),
                        nn.SiLU(inplace=True),
                        nn.Linear(h2, num_pollutants),
                    )
                    for _ in range(time_steps)
                ]
            )
            for field_mlp in self.field_mlps:
                nn.init.constant_(field_mlp[-1].bias, -1.5)
        else:
            self.c0_mlp = nn.Sequential(
                nn.Linear(field_decode_in, h1),
                nn.SiLU(inplace=True),
                nn.Linear(h1, h2),
                nn.SiLU(inplace=True),
                nn.Linear(h2, num_pollutants),
            )
            nn.init.zeros_(self.c0_mlp[-1].weight)
            nn.init.zeros_(self.c0_mlp[-1].bias)
            self.field_mlps = None

        self.lag_film = nn.Linear(time_context_dim, 2 * out_channels)
        self.temporal_conv = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(time_steps, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(_resolve_gn_groups(out_channels), out_channels)
        self.temporal_act = nn.SiLU(inplace=True)
        refine_layers: list[nn.Module] = []
        for _ in range(max(0, int(temporal_layers) - 1)):
            refine_layers.extend(
                [
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_resolve_gn_groups(out_channels), out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
        self.temporal_refine = nn.Sequential(*refine_layers)

    @staticmethod
    def _grid_coords(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Normalized grid in [-1, 1] as (x, y) per PyTorch grid_sample / pos encoding."""
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(h * w, 2)

    @staticmethod
    def _grid_coords_cells(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Grid coordinates in cell units as (x, y): [H*W, 2]."""
        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(h * w, 2)

    @staticmethod
    def _norm_coords(coord_yx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Map station coord [y, x] pixels to normalized (x, y) in [-1, 1]."""
        denom_y = max(h - 1, 1)
        denom_x = max(w - 1, 1)
        y = coord_yx[..., 0].to(dtype=torch.float32)
        x = coord_yx[..., 1].to(dtype=torch.float32)
        ny = y / float(denom_y) * 2.0 - 1.0
        nx = x / float(denom_x) * 2.0 - 1.0
        return torch.stack([nx, ny], dim=-1)

    def _pe_grid_coords(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.pe_mode == "cells":
            return self._grid_coords_cells(h, w, device, dtype)
        return self._grid_coords(h, w, device, dtype)

    def _pe_station_coords(self, coord_yx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if self.pe_mode == "cells":
            y = coord_yx[..., 0].to(dtype=torch.float32)
            x = coord_yx[..., 1].to(dtype=torch.float32)
            return torch.stack([x, y], dim=-1)
        return self._norm_coords(coord_yx, h, w)

    @staticmethod
    def _grid_station_dist2(coord_yx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Squared grid-cell distance between every grid pixel and station: [B, H*W, N]."""
        device = coord_yx.device
        sy = coord_yx[..., 0].to(dtype=torch.float32)  # [B, N]
        sx = coord_yx[..., 1].to(dtype=torch.float32)
        gy = torch.arange(h, device=device, dtype=torch.float32)
        gx = torch.arange(w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid_y = grid_y.reshape(1, h * w, 1)
        grid_x = grid_x.reshape(1, h * w, 1)
        dy = grid_y - sy.unsqueeze(1)
        dx = grid_x - sx.unsqueeze(1)
        return dy.square() + dx.square()

    @staticmethod
    def _topk_station_selection(
        coord_yx: torch.Tensor,
        station_mask: torch.Tensor,
        h: int,
        w: int,
        k: int,
        query_chunk: int = 16384,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest-k station indices and squared distances per grid pixel.

        Returns (topk_idx [B, H*W, k] long, dist2_topk [B, H*W, k] float32).
        Computed in query chunks so the full [B, H*W, N] matrix is never
        materialized at full-domain scale.
        """
        device = coord_yx.device
        b, n = station_mask.shape
        sy = coord_yx[..., 0].to(dtype=torch.float32)  # [B, N]
        sx = coord_yx[..., 1].to(dtype=torch.float32)
        pad = station_mask <= 0.5
        gy_all = torch.arange(h, device=device, dtype=torch.float32)
        gx_all = torch.arange(w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy_all, gx_all, indexing="ij")
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)
        lq = h * w
        k = min(int(k), n)

        idx_chunks: list[torch.Tensor] = []
        d2_chunks: list[torch.Tensor] = []
        inf = torch.tensor(float("inf"), device=device, dtype=torch.float32)
        for start in range(0, lq, query_chunk):
            end = min(lq, start + query_chunk)
            dy = grid_y[start:end].view(1, -1, 1) - sy.unsqueeze(1)  # [B, lc, N]
            dx = grid_x[start:end].view(1, -1, 1) - sx.unsqueeze(1)
            d2 = dy.square() + dx.square()
            d2 = torch.where(pad.unsqueeze(1), inf, d2)
            d2_k, idx_k = torch.topk(d2, k, dim=-1, largest=False)
            idx_chunks.append(idx_k)
            d2_chunks.append(d2_k)
        topk_idx = torch.cat(idx_chunks, dim=1)
        dist2_topk = torch.cat(d2_chunks, dim=1)
        # Padded stations selected only when fewer than k stations are active;
        # neutralize their inf distance (they are masked in attention anyway).
        dist2_topk = torch.where(torch.isfinite(dist2_topk), dist2_topk, torch.zeros_like(dist2_topk))
        return topk_idx, dist2_topk

    def _sample_grid_ctx(self, tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b = tensor.shape[0]
        coords = self._grid_coords(h, w, tensor.device, tensor.dtype)
        grid = coords.view(1, h, w, 2).expand(b, h, w, 2)
        sampled = F.grid_sample(
            tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.flatten(2).transpose(1, 2)

    def _sample_met_ctx(self, met_direct: torch.Tensor, met_temporal: torch.Tensor, lag: int) -> torch.Tensor:
        b, _, h, w = met_direct.shape
        met_lag = met_temporal[:, lag]
        met_cat = torch.cat([met_direct, met_lag], dim=1)
        return self._sample_grid_ctx(met_cat, h, w)

    def _sample_static_ctx(self, static_ctx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return self._sample_grid_ctx(static_ctx, h, w)

    def _compute_geometry(
        self,
        coord: torch.Tensor,
        station_mask: torch.Tensor,
        h: int,
        w: int,
    ) -> tuple:
        """Compute pe-offset-independent topk geometry (dist2, topk_idx, dist2_topk).

        The returned tuple can be passed as ``precomputed_geometry`` to ``forward``
        so the jitter-consistency second pass skips the expensive ``_topk_station_selection``.
        """
        n = station_mask.shape[1]
        dist2, topk_idx, dist2_topk = None, None, None
        use_topk = self.station_topk > 0 and n > self.station_topk
        if use_topk:
            topk_idx, dist2_topk = self._topk_station_selection(coord, station_mask, h, w, k=self.station_topk)
            if not self.distance_bias_enabled:
                dist2_topk = None
        elif self.distance_bias_enabled:
            dist2 = self._grid_station_dist2(coord, h, w)
        return dist2, topk_idx, dist2_topk

    def _sample_pe_offset(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        """Per-sample (x, y) PE origin offset, shared by grid and stations."""
        if not (self.pe_jitter_enabled and self.training):
            return None
        max_wl = getattr(self.pos_enc, "max_wavelength_cells", 0.0)
        if max_wl <= 0.0:
            return None
        return torch.rand(batch_size, 2, device=device, dtype=torch.float32) * float(max_wl)

    def _encode_station_tokens(
        self,
        coord: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        h: int,
        w: int,
        pe_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coord_pe = self._pe_station_coords(coord, h, w)
        if pe_offset is not None:
            coord_pe = coord_pe + pe_offset[:, None, :]
        pos = self.pos_enc(coord_pe)
        val = values * valid
        feat = torch.cat([val, valid, pos], dim=-1)
        return self.station_embed(feat)

    def _decode_lag_field(
        self,
        coord: torch.Tensor,
        station_mask: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        met_ctx: torch.Tensor,
        static_ctx: torch.Tensor | None,
        h: int,
        w: int,
        dist2: torch.Tensor | None = None,
        topk_idx: torch.Tensor | None = None,
        dist2_topk: torch.Tensor | None = None,
        pe_offset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = coord.shape[0]
        station_tokens = self._encode_station_tokens(coord, values, valid, h, w, pe_offset=pe_offset)
        key_padding_mask = station_mask <= 0.5

        grid_pe = self._pe_grid_coords(h, w, coord.device, torch.float32).unsqueeze(0).expand(b, h * w, 2)
        if pe_offset is not None:
            grid_pe = grid_pe + pe_offset[:, None, :]
        grid_pos = self.pos_enc(grid_pe)
        ctx_parts = [grid_pos, met_ctx]
        if static_ctx is not None:
            ctx_parts.append(static_ctx)
        query_in = torch.cat(ctx_parts, dim=-1)
        query = self.query_proj(query_in)

        has_active = (station_mask > 0.5).any(dim=1)
        if has_active.all():
            x = query
            for block in self.cross_blocks:
                x = block(
                    x, station_tokens, station_tokens, key_padding_mask=key_padding_mask,
                    dist2=dist2, topk_idx=topk_idx, dist2_topk=dist2_topk,
                )
        elif has_active.any():
            x_attn = query
            for block in self.cross_blocks:
                x_attn = block(
                    x_attn, station_tokens, station_tokens, key_padding_mask=key_padding_mask,
                    dist2=dist2, topk_idx=topk_idx, dist2_topk=dist2_topk,
                )
            x = torch.where(has_active.view(b, 1, 1), x_attn, query)
        else:
            x = query

        ctx_tail = query_in[:, :, self.pos_enc.out_dim :]
        decoded = torch.cat([x, ctx_tail], dim=-1)
        field_decoded = torch.cat([x, grid_pos], dim=-1)
        out = self.out_mlp(decoded)
        feat = out.transpose(1, 2).reshape(b, self.out_channels, h, w)
        return feat, decoded, field_decoded, met_ctx

    def forward(
        self,
        coord: torch.Tensor,
        station_mask: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        met_temporal_10x3: torch.Tensor,
        met_direct_21: torch.Tensor,
        time_context_lag: torch.Tensor | None = None,
        static_ctx: torch.Tensor | None = None,
        return_field_lags: bool = False,
        precomputed_geometry: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if coord.dim() != 3 or coord.shape[-1] != 2:
            raise ValueError(f"coord must be [B,N,2], got {tuple(coord.shape)}")
        if values.shape != valid.shape:
            raise ValueError(f"values/valid shape mismatch: {tuple(values.shape)} vs {tuple(valid.shape)}")
        if values.dim() != 4:
            raise ValueError(f"values must be [B,N,T,P], got {tuple(values.shape)}")

        b, n, t_steps, p = values.shape
        if t_steps != self.time_steps or p != self.num_pollutants:
            raise ValueError(
                f"values must be [B,N,{self.time_steps},{self.num_pollutants}], got {tuple(values.shape)}"
            )

        h, w = met_direct_21.shape[-2:]
        static_grid_ctx: torch.Tensor | None = None
        if self.static_ctx_enabled and static_ctx is not None:
            if static_ctx.dim() != 4:
                raise ValueError(f"static_ctx must be [B,C,H,W], got {tuple(static_ctx.shape)}")
            static_grid_ctx = self._sample_static_ctx(static_ctx, h, w)

        if precomputed_geometry is not None:
            dist2, topk_idx, dist2_topk = precomputed_geometry
        else:
            dist2, topk_idx, dist2_topk = self._compute_geometry(coord, station_mask, h, w)

        # One shared PE origin offset per sample, identical across lags so the
        # temporal stack stays geometrically consistent.
        pe_offset = self._sample_pe_offset(b, coord.device)

        lag_feats: list[torch.Tensor] = []
        field_lags: list[torch.Tensor] = []
        c0_inr: torch.Tensor | None = None
        for lag in range(self.time_steps):
            met_ctx = self._sample_met_ctx(met_direct_21, met_temporal_10x3, lag=lag)
            feat, decoded, field_decoded, _ = self._decode_lag_field(
                coord=coord,
                station_mask=station_mask,
                values=values[:, :, lag, :],
                valid=valid[:, :, lag, :],
                met_ctx=met_ctx,
                static_ctx=static_grid_ctx,
                h=h,
                w=w,
                dist2=dist2,
                topk_idx=topk_idx,
                dist2_topk=dist2_topk,
                pe_offset=pe_offset,
            )
            if self.field_heads and self.field_mlps is not None:
                field_out = self.field_mlps[lag](field_decoded)
                field_map = F.softplus(
                    field_out.transpose(1, 2).reshape(b, self.num_pollutants, h, w)
                )
                field_lags.append(field_map)
                if lag == 0:
                    c0_inr = field_map
            else:
                if lag == 0:
                    c0_out = self.c0_mlp(field_decoded)
                    c0_inr = F.softplus(
                        c0_out.transpose(1, 2).reshape(b, self.num_pollutants, h, w)
                    )
                    field_lags.append(c0_inr)
            if time_context_lag is not None:
                feat = _apply_film_2d(feat, self.lag_film, time_context_lag[:, lag])
            lag_feats.append(feat)

        stacked = torch.stack(lag_feats, dim=2)
        out = self.temporal_act(self.temporal_norm(self.temporal_conv(stacked).squeeze(2)))
        out = self.temporal_refine(out)
        if c0_inr is None:
            raise RuntimeError("c0_inr was not computed for lag=0")

        if return_field_lags:
            if not field_lags:
                field_lags = [c0_inr]
            while len(field_lags) < self.time_steps:
                field_lags.append(torch.zeros_like(c0_inr))
            field_lags_tensor = torch.stack(field_lags, dim=1)
            return out, c0_inr, field_lags_tensor

        return out, c0_inr


def scatter_advection_c0_from_stations(
    coord: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
    station_mask: torch.Tensor,
    hw: Tuple[int, int],
    normalizer,
    lag_index: int = 0,
) -> torch.Tensor:
    """Build normalized obs raster [B,6,H,W] for advection loss only."""
    import numpy as np

    b, n, t_steps, p = values.shape
    h, w = hw
    device = values.device
    dtype = values.dtype
    out = torch.zeros((b, p, h, w), device=device, dtype=dtype)

    values_np = values.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    coord_np = coord.detach().cpu().numpy()
    mask_np = station_mask.detach().cpu().numpy()

    for bi in range(b):
        for si in range(n):
            if mask_np[bi, si] <= 0.5:
                continue
            y = int(coord_np[bi, si, 0])
            x = int(coord_np[bi, si, 1])
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            raw = values_np[bi, si, lag_index, :].copy()
            m = valid_np[bi, si, lag_index, :] > 0.5
            if not np.any(m):
                continue
            raw[~m] = 0.0
            norm = normalizer.normalize_obs_t_pollutants_6(raw)
            out[bi, :, y, x] = torch.from_numpy(norm).to(device=device, dtype=dtype)

    return out


__all__ = [
    "ObsSparseINREncoder",
    "resolve_inr_encoder_config",
    "scatter_advection_c0_from_stations",
]

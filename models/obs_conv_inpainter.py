"""Conv-based field inpainter with NW-prior cross-attention + FiLM decoder (v4conv).

Design principles:
  - Obs draws the field. Met only modulates HOW obs information spreads.
  - Met never appears as spatial input — it cannot paint the field directly.
  - NW kernel prior guarantees ALL pixels receive station info (structurally dense).
  - V = raw station values (per-species) → attention output is direct interpolation.
  - Learned Q@K breaks isotropy: wind, terrain dependence in weights.
  - Conv decoder adds spatial coherence (smooth transitions, met-conditioned detail).

Architecture:
  1. Station K encoder: MLP(value, valid) → per-station key features [B, N, D]
  2. V = station values (species-explicit): [B, N, P]
  3. NW prior: -||grid - station||² / (2σ²) → attention bias [B, H, HW', N]
  4. Cross-attention: Q(grid) @ K(station) + β*nw_bias → weights
     → output = mean_heads(weights) @ V → [B, HW', P] (species-explicit interp)
  5. Conv decoder: concat(interp, attn_feat) → FiLM blocks → field [B, P, H, W]
"""

from __future__ import annotations

from typing import Any, Mapping

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_conv_inpainter_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        config = {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    inr_cfg = model_cfg.get("inr", {}) if isinstance(model_cfg.get("inr"), Mapping) else {}
    inp_cfg = inr_cfg.get("conv_inpainter", {}) if isinstance(inr_cfg.get("conv_inpainter"), Mapping) else {}
    return {
        "num_pollutants": int(inr_cfg.get("num_pollutants", 6)),
        "time_steps": int(inr_cfg.get("time_steps", 3)),
        "met_ctx_channels": int(inr_cfg.get("met_ctx_channels", 31)),
        "static_ctx_channels": int(inr_cfg.get("static_ctx_channels", 10)),
        "static_ctx_enabled": bool(inr_cfg.get("static_ctx_enabled", False)),
        "hidden_channels": int(inp_cfg.get("hidden_channels", 96)),
        "num_blocks": int(inp_cfg.get("num_blocks", 6)),
        "kernel_size": int(inp_cfg.get("kernel_size", 5)),
        "coarse_sigma": float(inp_cfg.get("coarse_sigma", 30.0)),
        "film_hidden": int(inp_cfg.get("film_hidden", 128)),
        "attn_heads": int(inp_cfg.get("attn_heads", 8)),
        "attn_dim": int(inp_cfg.get("attn_dim", 64)),
        "attn_downsample": int(inp_cfg.get("attn_downsample", 4)),
        "nw_sigma_init": float(inp_cfg.get("nw_sigma_init", 8.0)),
        # Species differentiation (both identity at init → safe warm-start):
        #   species_head_mix: per-species softmax mixture over attention heads
        #     (zero logits == uniform == legacy weights.mean(heads)).
        #   species_decoder_head: zero-init per-species residual readout added
        #     to the shared 1x1 head (adds nothing at init).
        "species_head_mix": bool(inp_cfg.get("species_head_mix", False)),
        "species_decoder_head": bool(inp_cfg.get("species_decoder_head", False)),
        "species_decoder_hidden": int(inp_cfg.get("species_decoder_hidden", 32)),
    }


# ---------------------------------------------------------------------------
# FiLM modules
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """Generate per-layer FiLM parameters (gamma, beta) from met context."""

    def __init__(self, met_channels: int, film_hidden: int, target_channels: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.target_channels = target_channels
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(met_channels, film_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(film_hidden, film_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(film_hidden, num_layers * target_channels * 2),
        )

    def forward(self, met: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """met: [B, C, H, W] → list of (gamma [B,Ch,1,1], beta [B,Ch,1,1]) per layer."""
        params = self.encoder(met)
        b = params.shape[0]
        params = params.view(b, self.num_layers, self.target_channels, 2)
        out = []
        for i in range(self.num_layers):
            gamma = params[:, i, :, 0].view(b, self.target_channels, 1, 1) + 1.0
            beta = params[:, i, :, 1].view(b, self.target_channels, 1, 1)
            out.append((gamma, beta))
        return out


class FiLMResBlock(nn.Module):
    """Conv residual block with FiLM conditioning."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad, dilation=dilation, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad, dilation=dilation, padding_mode="replicate")
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = h * gamma + beta
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


# ---------------------------------------------------------------------------
# NW-prior cross-attention with species-explicit V
# ---------------------------------------------------------------------------

class StationKeyEncoder(nn.Module):
    """Encode (value, valid) per station → key features for attention."""

    def __init__(self, num_pollutants: int, out_dim: int) -> None:
        super().__init__()
        in_dim = num_pollutants * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """values [B,N,P], valid [B,N,P] → [B, N, D]."""
        x = torch.cat([values * valid, valid], dim=-1)
        return self.mlp(x)


class NWCrossAttentionExplicitV(nn.Module):
    """Multi-head cross-attention with NW prior and species-explicit value path.

    Key insight: V = raw station obs values [B, N, P].
    Attention weights are shared across species, but since V is per-species,
    the output is automatically species-differentiated interpolation.

    Additionally outputs learned features from a separate V_feat path
    for the conv decoder to use.
    """

    def __init__(
        self,
        grid_dim: int,
        station_dim: int,
        num_pollutants: int,
        num_heads: int = 8,
        nw_sigma_init: float = 8.0,
        species_head_mix: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = grid_dim // num_heads
        self.num_pollutants = num_pollutants
        assert grid_dim % num_heads == 0

        self.q_proj = nn.Linear(grid_dim, grid_dim)
        self.k_proj = nn.Linear(station_dim, grid_dim)
        # Separate V for learned features (conv decoder input)
        self.v_feat_proj = nn.Linear(station_dim, grid_dim)

        # Per-head learnable sigma for NW prior
        self.log_sigma = nn.Parameter(
            torch.full((num_heads,), math.log(nw_sigma_init))
        )
        # Learnable strength of NW prior
        self._nw_beta_raw = nn.Parameter(torch.tensor(0.0))

        # Per-species mixture over heads. Zero logits → softmax uniform →
        # numerically identical to legacy weights.mean(dim=1); checkpoints
        # without this key warm-start at the identity point (strict=False).
        self.species_head_mix = bool(species_head_mix)
        if self.species_head_mix:
            self.species_mix_logits = nn.Parameter(
                torch.zeros(num_pollutants, num_heads)
            )
        else:
            self.species_mix_logits = None

    def forward(
        self,
        grid_feat: torch.Tensor,
        station_feat: torch.Tensor,
        station_values: torch.Tensor,
        station_valid: torch.Tensor,
        nw_dist2: torch.Tensor,
        station_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        grid_feat: [B, HW, D]
        station_feat: [B, N, D] — encoded station keys
        station_values: [B, N, P] — raw obs values
        station_valid: [B, N, P] — validity mask
        nw_dist2: [B, HW, N] — squared distances
        station_mask: [B, N] — 1=active station
        Returns:
            interp_values: [B, HW, P] — species-explicit interpolated field
            feat_out: [B, HW, D] — learned features for conv decoder
        """
        b, hw, d = grid_feat.shape
        n = station_feat.shape[1]
        h = self.num_heads
        dk = self.head_dim
        p = self.num_pollutants

        Q = self.q_proj(grid_feat).view(b, hw, h, dk).permute(0, 2, 1, 3)  # [B, H, HW, dk]
        K = self.k_proj(station_feat).view(b, n, h, dk).permute(0, 2, 1, 3)  # [B, H, N, dk]

        # Attention logits
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)  # [B, H, HW, N]

        # NW prior: β * (-dist² / (2σ²)) per head
        sigma2 = 2.0 * self.log_sigma.exp().square()  # [H]
        nw_bias = -nw_dist2.unsqueeze(1) / sigma2.view(1, h, 1, 1)  # [B, H, HW, N]
        beta = F.softplus(self._nw_beta_raw) + 0.5

        attn_logits = attn_logits + beta * nw_bias

        # Mask padded stations
        pad_mask = (station_mask < 0.5).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn_logits = attn_logits.masked_fill(pad_mask, -1e9)

        weights = F.softmax(attn_logits, dim=-1)  # [B, H, HW, N]

        # Zero out invalid species in values before interpolation
        v_masked = station_values * station_valid  # [B, N, P]
        if self.species_mix_logits is not None:
            # Per-species head mixture: each species interpolates with its own
            # convex combination of the head kernels (own effective sigma /
            # anisotropy). Uniform mixture == legacy mean over heads.
            # Mix per-head interpolations (cheap [B,H,HW,P]) instead of
            # per-species weights ([B,P,HW,N] would dominate peak memory).
            mix = F.softmax(self.species_mix_logits, dim=-1)  # [P, H]
            head_interp = torch.einsum("bhqn,bnp->bhqp", weights, v_masked)
            head_valid = torch.einsum("bhqn,bnp->bhqp", weights, station_valid)
            interp_values = torch.einsum("ph,bhqp->bqp", mix, head_interp)  # [B, HW, P]
            valid_w = torch.einsum("ph,bhqp->bqp", mix, head_valid)  # [B, HW, P]
        else:
            # Legacy: single head-averaged kernel shared by all species.
            avg_weights = weights.mean(dim=1)  # [B, HW, N]
            valid_w = torch.bmm(avg_weights, station_valid)  # [B, HW, P]
            interp_values = torch.bmm(avg_weights, v_masked)  # [B, HW, P]
        interp_values = interp_values / valid_w.clamp_min(1e-6)  # normalize by valid weight sum

        # Learned feature path (for conv decoder context)
        V_feat = self.v_feat_proj(station_feat).view(b, n, h, dk).permute(0, 2, 1, 3)
        feat_out = torch.matmul(weights, V_feat)  # [B, H, HW, dk]
        feat_out = feat_out.permute(0, 2, 1, 3).reshape(b, hw, d)  # [B, HW, D]

        return interp_values, feat_out


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class ConvInpainterEncoder(nn.Module):
    """NW-prior cross-attention (species-explicit V) → FiLM conv decoder → dense field."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        resolved = resolve_conv_inpainter_config(config)
        self.num_pollutants = int(resolved["num_pollutants"])
        self.time_steps = int(resolved["time_steps"])
        self.met_ctx_channels = int(resolved["met_ctx_channels"])
        self.static_ctx_channels = int(resolved["static_ctx_channels"])
        self.static_ctx_enabled = bool(resolved["static_ctx_enabled"])
        hidden = int(resolved["hidden_channels"])
        ks = int(resolved["kernel_size"])
        film_hidden = int(resolved["film_hidden"])
        self.attn_downsample = int(resolved["attn_downsample"])

        attn_heads = int(resolved["attn_heads"])
        attn_dim = int(resolved["attn_dim"])
        nw_sigma_init = float(resolved["nw_sigma_init"])

        # Station key encoder
        self.station_enc = StationKeyEncoder(self.num_pollutants, attn_dim)

        # Grid positional embedding (learned, at reduced resolution)
        self.grid_pos_embed = nn.Parameter(torch.randn(1, attn_dim, 48, 48) * 0.02)

        # Cross-attention with species-explicit V
        self.cross_attn = NWCrossAttentionExplicitV(
            grid_dim=attn_dim,
            station_dim=attn_dim,
            num_pollutants=self.num_pollutants,
            num_heads=attn_heads,
            nw_sigma_init=nw_sigma_init,
            species_head_mix=bool(resolved["species_head_mix"]),
        )
        self.attn_norm = nn.LayerNorm(attn_dim)

        # Progressive decoder input: interp_values (P) + attn_feat (D)
        decoder_in = self.num_pollutants + attn_dim
        if self.static_ctx_enabled:
            decoder_in += self.static_ctx_channels

        self.scale0_stem = nn.Conv2d(decoder_in, hidden, 3, padding=1, padding_mode="replicate")
        self.scale0_blocks = nn.ModuleList([FiLMResBlock(hidden, ks, 1) for _ in range(2)])
        self.scale1_blocks = nn.ModuleList([FiLMResBlock(hidden, ks, 1) for _ in range(2)])
        self.scale2_blocks = nn.ModuleList([FiLMResBlock(hidden, ks, 1) for _ in range(2)])

        self.head = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.num_pollutants, 1),
        )

        # Per-species residual readout: 3x3 spatial kernels give each species
        # its own decoding of the shared features (the 1x1 head alone makes all
        # species linear combos of one basis → same shapes, different scale).
        # Last layer zero-init → contributes nothing at init / warm-start.
        self.species_decoder_head = bool(resolved["species_decoder_head"])
        if self.species_decoder_head:
            sp_hidden = int(resolved["species_decoder_hidden"])
            self.species_head = nn.Sequential(
                nn.GroupNorm(8, hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, sp_hidden, 3, padding=1, padding_mode="replicate"),
                nn.SiLU(inplace=True),
                nn.Conv2d(sp_hidden, self.num_pollutants, 3, padding=1, padding_mode="replicate"),
            )
            nn.init.zeros_(self.species_head[-1].weight)
            nn.init.zeros_(self.species_head[-1].bias)
        else:
            self.species_head = None

        # Met FiLM generator — 6 total blocks (2 per scale)
        met_in = self.met_ctx_channels
        self.film_gen = FiLMGenerator(met_in, film_hidden, hidden, 6)

    @staticmethod
    def _upsample_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _get_grid_features(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Get positional grid features at reduced resolution [1, D, h', w']."""
        ds = self.attn_downsample
        h_ds = h // ds
        w_ds = w // ds
        pos = F.interpolate(self.grid_pos_embed, size=(h_ds, w_ds), mode="bilinear", align_corners=False)
        return pos.to(device)

    def _compute_nw_dist2(
        self,
        coord: torch.Tensor,
        station_mask: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Squared distances from grid pixels (reduced res) to stations."""
        ds = self.attn_downsample
        h_ds = h // ds
        w_ds = w // ds
        b, n, _ = coord.shape
        device = coord.device

        sy = coord[..., 0].float() / ds
        sx = coord[..., 1].float() / ds

        gy = torch.arange(h_ds, device=device, dtype=torch.float32)
        gx = torch.arange(w_ds, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid_y = grid_y.reshape(1, h_ds * w_ds, 1)
        grid_x = grid_x.reshape(1, h_ds * w_ds, 1)

        dy = grid_y - sy.unsqueeze(1)
        dx = grid_x - sx.unsqueeze(1)
        return dy.square() + dx.square()

    def _build_met_features(
        self,
        met_direct_21: torch.Tensor,
        met_temporal_10x3: torch.Tensor,
        lag: int,
    ) -> torch.Tensor:
        met_lag = met_temporal_10x3[:, lag]
        return torch.cat([met_direct_21, met_lag], dim=1)

    def _forward_single_lag(
        self,
        coord: torch.Tensor,
        values_lag: torch.Tensor,
        valid_lag: torch.Tensor,
        station_mask: torch.Tensor,
        met_direct_21: torch.Tensor,
        met_temporal_10x3: torch.Tensor,
        lag: int,
        h: int,
        w: int,
        static_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = coord.shape[0]
        ds = self.attn_downsample
        h_ds = h // ds
        w_ds = w // ds

        # 1. Encode station keys
        station_feat = self.station_enc(values_lag, valid_lag)  # [B, N, D]

        # 2. Grid positional features (Q source)
        grid_pos = self._get_grid_features(h, w, coord.device)
        grid_feat = grid_pos.expand(b, -1, -1, -1)
        grid_feat_flat = grid_feat.flatten(2).permute(0, 2, 1)  # [B, HW', D]

        # 3. NW distance prior
        nw_dist2 = self._compute_nw_dist2(coord, station_mask, h, w)

        # 4. Cross-attention: species-explicit interpolation + learned features
        interp_values, feat_out = self.cross_attn(
            grid_feat_flat, station_feat,
            values_lag, valid_lag,
            nw_dist2, station_mask,
        )
        # interp_values: [B, HW', P] — weighted average of station obs per species
        # feat_out: [B, HW', D] — learned context features

        # Residual + norm on feat
        feat_out = self.attn_norm(feat_out + grid_feat_flat)

        # 5. Reshape to spatial
        interp_map = interp_values.permute(0, 2, 1).reshape(b, self.num_pollutants, h_ds, w_ds)
        feat_map = feat_out.permute(0, 2, 1).reshape(b, -1, h_ds, w_ds)

        # 6. Decoder input: species-explicit interp + learned features
        parts = [interp_map, feat_map]
        if self.static_ctx_enabled and static_ctx is not None:
            static_ds = F.interpolate(static_ctx, size=(h_ds, w_ds), mode="bilinear", align_corners=False)
            parts.append(static_ds)
        dense_in = torch.cat(parts, dim=1)

        # 7. Met → FiLM
        met_feat = self._build_met_features(met_direct_21, met_temporal_10x3, lag)
        film_params = self.film_gen(met_feat)
        fp = iter(film_params)

        # Intermediate sizes
        h_mid = (h_ds + h) // 2
        w_mid = (w_ds + w) // 2

        # 8. Progressive decode
        x = self.scale0_stem(dense_in)
        for block in self.scale0_blocks:
            gamma, beta = next(fp)
            x = block(x, gamma, beta)

        x = self._upsample_to(x, (h_mid, w_mid))
        for block in self.scale1_blocks:
            gamma, beta = next(fp)
            x = block(x, gamma, beta)

        x = self._upsample_to(x, (h, w))
        for block in self.scale2_blocks:
            gamma, beta = next(fp)
            x = block(x, gamma, beta)

        # 9. Head: residual from upsampled interpolation
        interp_full = self._upsample_to(interp_map, (h, w))  # [B, P, H, W]
        field = self.head(x) + interp_full
        if self.species_head is not None:
            field = field + self.species_head(x)

        return F.leaky_relu(field, negative_slope=0.01)

    def forward_field_lags(
        self,
        coord: torch.Tensor,
        station_mask: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        met_temporal_10x3: torch.Tensor,
        met_direct_21: torch.Tensor,
        static_ctx: torch.Tensor | None = None,
        precomputed_geometry: tuple | None = None,
        return_decomposition: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode all lags → [B, L, P, H, W]."""
        if coord.dim() != 3 or coord.shape[-1] != 2:
            raise ValueError(f"coord must be [B,N,2], got {tuple(coord.shape)}")
        if values.dim() != 4:
            raise ValueError(f"values must be [B,N,T,P], got {tuple(values.shape)}")

        h, w = met_direct_21.shape[-2:]
        fields: list[torch.Tensor] = []

        for lag in range(self.time_steps):
            field = self._forward_single_lag(
                coord,
                values[:, :, lag, :],
                valid[:, :, lag, :],
                station_mask,
                met_direct_21,
                met_temporal_10x3,
                lag,
                h,
                w,
                static_ctx=static_ctx,
            )
            fields.append(field)

        field_lags = torch.stack(fields, dim=1)
        if not return_decomposition:
            return field_lags
        h00 = field_lags[:, 0]
        decomp = {
            "inr_baseline_raw": h00,
            "inr_kernel_field": h00,
            "inr_residual_delta": torch.zeros_like(h00),
        }
        return field_lags, decomp

    def forward_h00_only(
        self,
        coord: torch.Tensor,
        station_mask: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        met_temporal_10x3: torch.Tensor,
        met_direct_21: torch.Tensor,
        static_ctx: torch.Tensor | None = None,
        precomputed_geometry: tuple | None = None,
        return_decomposition: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode lag=0 only → [B, P, H, W]."""
        if coord.dim() != 3 or coord.shape[-1] != 2:
            raise ValueError(f"coord must be [B,N,2], got {tuple(coord.shape)}")
        if values.dim() != 4:
            raise ValueError(f"values must be [B,N,T,P], got {tuple(values.shape)}")

        h, w = met_direct_21.shape[-2:]
        field = self._forward_single_lag(
            coord,
            values[:, :, 0, :],
            valid[:, :, 0, :],
            station_mask,
            met_direct_21,
            met_temporal_10x3,
            0,
            h,
            w,
            static_ctx=static_ctx,
        )
        if not return_decomposition:
            return field
        decomp = {
            "inr_baseline_raw": field,
            "inr_kernel_field": field,
            "inr_residual_delta": torch.zeros_like(field),
        }
        return field, decomp


__all__ = ["ConvInpainterEncoder", "resolve_conv_inpainter_config"]

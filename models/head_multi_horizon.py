from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_blocks import ConvNormAct


class HeadMultiHorizon(nn.Module):
    """Residual + delta cumulative multi-horizon regression head.

    Optional head-level horizon FiLM:
    - Uses per-horizon valid-time context (t+h) plus log1p(h) lead-time scale.
    - Keeps interface backward-compatible when time_context_horizon is omitted.

    Optional quantile output (quantile_levels=(lo, 0.5, hi)):
    - Internally predicts (median, lo_offset_raw, hi_offset_raw) per pollutant.
    - Non-crossing by construction: lo = med - softplus(raw), hi = med + softplus(raw).
    - Default forward output stays the median: [B, 36, H, W], so downstream
      validate/predict paths are unchanged. Full quantiles are returned only
      when return_quantiles=True as [B, Hh, P, K, H, W] (K ordered lo/med/hi).

    Default behavior:
    - Input: [B, 72, H, W]
    - Output (flat): [B, 36, H, W]  where 36 = 6 horizons x 6 pollutants
    - Output (reshaped): [B, 6, 6, H, W] = [B, Hh, P, H, W]
    """

    def __init__(
        self,
        in_channels: int = 72,
        num_horizons: int = 6,
        num_pollutants: int = 6,
        use_pre_head: bool = True,
        pre_head_layers: int = 1,
        time_context_dim: int = 512,
        horizon_hours: tuple[int, ...] | None = None,
        conv_padding_mode: str = "replicate",
        quantile_levels: tuple[float, ...] | None = None,
        init_quantile_offset: float = 0.05,
    ) -> None:
        super().__init__()
        if pre_head_layers < 0:
            raise ValueError(f"pre_head_layers must be >= 0, got {pre_head_layers}")

        self.in_channels = in_channels
        self.num_horizons = num_horizons
        self.num_pollutants = num_pollutants
        self.out_channels = num_horizons * num_pollutants
        self.time_context_dim = int(time_context_dim)

        if quantile_levels is not None:
            levels = tuple(float(q) for q in quantile_levels)
            if len(levels) != 3 or not (0.0 < levels[0] < 0.5 < levels[2] < 1.0) or levels[1] != 0.5:
                raise ValueError(
                    f"quantile_levels must be (lo, 0.5, hi) with 0<lo<0.5<hi<1, got {levels}"
                )
            self.quantile_levels: tuple[float, ...] | None = levels
            self.num_components = 3  # (median, lo_offset_raw, hi_offset_raw)
        else:
            self.quantile_levels = None
            self.num_components = 1

        blocks: list[nn.Module] = []
        if use_pre_head and pre_head_layers > 0:
            for _ in range(pre_head_layers):
                blocks.append(
                    ConvNormAct(in_channels, in_channels, kernel_size=3, padding_mode=conv_padding_mode)
                )
        self.pre_head = nn.Sequential(*blocks) if blocks else nn.Identity()

        # Base field shared across horizons + per-horizon delta residuals.
        proj_out = self.num_pollutants * self.num_components
        self.base_proj = nn.Conv2d(in_channels, proj_out, kernel_size=1, bias=True)
        self.delta_proj = nn.Conv2d(in_channels, proj_out, kernel_size=1, bias=True)

        if self.num_components == 3:
            # Start with a small (~init_quantile_offset) symmetric interval.
            raw0 = math.log(math.expm1(max(1e-4, float(init_quantile_offset))))
            with torch.no_grad():
                bias = self.base_proj.bias.view(self.num_pollutants, self.num_components)
                bias[:, 1] = raw0
                bias[:, 2] = raw0
                dbias = self.delta_proj.bias.view(self.num_pollutants, self.num_components)
                dbias[:, 1] = 0.0
                dbias[:, 2] = 0.0

        if horizon_hours is None:
            horizon_hours = (1, 2, 4, 8, 12, 24) if num_horizons == 6 else tuple(range(1, num_horizons + 1))
        if len(horizon_hours) != num_horizons:
            raise ValueError(
                f"horizon_hours length mismatch: expected {num_horizons}, got {len(horizon_hours)}"
            )
        self.register_buffer("_horizon_hours", torch.tensor(horizon_hours, dtype=torch.float32), persistent=False)

        cond_dim = self.time_context_dim + 1  # valid-time context + log1p(h)
        hidden = max(32, in_channels)
        self.horizon_film = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * in_channels),
        )
        # Start close to identity modulation: gamma ~= 1, beta ~= 0.
        nn.init.zeros_(self.horizon_film[-1].weight)
        nn.init.zeros_(self.horizon_film[-1].bias)

    def _lead_time_encoding(self, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
        h = self._horizon_hours.to(device=device, dtype=dtype)
        logh = torch.log1p(h).view(1, self.num_horizons, 1).expand(batch_size, self.num_horizons, 1)
        return logh

    def forward(
        self,
        x: torch.Tensor,
        reshape_output: bool = False,
        time_context_horizon: torch.Tensor | None = None,
        time_context: torch.Tensor | None = None,
        return_quantiles: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Head input must be [B,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Head input channels mismatch: expected {self.in_channels}, got {x.shape[1]}"
            )

        feat = self.pre_head(x)
        b, _, h, w = feat.shape

        if time_context_horizon is None and time_context is not None:
            time_context_horizon = time_context[:, None, :].expand(b, self.num_horizons, self.time_context_dim)

        if time_context_horizon is not None:
            if time_context_horizon.dim() != 3 or time_context_horizon.shape[0] != b:
                raise ValueError(
                    "time_context_horizon must be [B,H,D] with same B as head input, "
                    f"got shape={tuple(time_context_horizon.shape)} for B={b}"
                )
            if time_context_horizon.shape[1] != self.num_horizons:
                raise ValueError(
                    f"time_context_horizon H mismatch: expected {self.num_horizons}, "
                    f"got {time_context_horizon.shape[1]}"
                )
            if time_context_horizon.shape[2] != self.time_context_dim:
                raise ValueError(
                    f"time_context_horizon dim mismatch: expected {self.time_context_dim}, "
                    f"got {time_context_horizon.shape[2]}"
                )
            lead_enc = self._lead_time_encoding(device=feat.device, dtype=feat.dtype, batch_size=b)
            cond = torch.cat(
                [time_context_horizon.to(dtype=feat.dtype), lead_enc],
                dim=-1,
            )
            film = self.horizon_film(cond.reshape(b * self.num_horizons, -1)).view(
                b, self.num_horizons, 2, self.in_channels
            )
            dgamma = film[:, :, 0, :]
            beta = film[:, :, 1, :]
            gamma = 1.0 + dgamma
            feat_h = feat.unsqueeze(1) * gamma[..., None, None] + beta[..., None, None]
        else:
            feat_h = feat.unsqueeze(1).expand(b, self.num_horizons, self.in_channels, h, w)

        k = self.num_components
        base = self.base_proj(feat).view(b, 1, self.num_pollutants, k, h, w)
        delta_flat = self.delta_proj(feat_h.reshape(b * self.num_horizons, self.in_channels, h, w))
        delta = delta_flat.view(b, self.num_horizons, self.num_pollutants, k, h, w)

        residual = torch.cumsum(delta, dim=1)
        y_raw = base + residual  # [B, Hh, P, K, H, W]

        if k == 1:
            med = y_raw[:, :, :, 0]
            quantiles = None
        else:
            med = y_raw[:, :, :, 0]
            lo = med - F.softplus(y_raw[:, :, :, 1])
            hi = med + F.softplus(y_raw[:, :, :, 2])
            quantiles = torch.stack([lo, med, hi], dim=3)

        out = med if reshape_output else med.reshape(b, self.out_channels, h, w)
        if return_quantiles:
            if quantiles is None:
                quantiles = med.unsqueeze(3)
            return out, quantiles
        return out


__all__ = ["HeadMultiHorizon"]

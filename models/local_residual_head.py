from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_blocks import ConvNormAct, _resolve_gn_groups


class HorizonFiLM(nn.Module):
    """Shared FiLM: time_context + lead_time → modulated feat."""

    def __init__(self, feat_channels: int, time_context_dim: int) -> None:
        super().__init__()
        cond_dim = time_context_dim + 1
        self.feat_channels = feat_channels
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feat_channels),
            nn.SiLU(inplace=True),
            nn.Linear(feat_channels, 2 * feat_channels),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        with torch.no_grad():
            bias = self.mlp[-1].bias
            bias[:feat_channels].zero_()
            bias[feat_channels:].normal_(0, 0.01)

    def forward(self, cond: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        film = self.mlp(cond)
        dgamma, beta = torch.chunk(film, 2, dim=1)
        gamma = (1.0 + dgamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class HorizonHead(nn.Module):
    """Single-horizon decoder: [B, in_ch, H, W] → [B, P*nc, H, W]."""

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        num_pollutants: int,
        num_components: int,
        conv_padding_mode: str = "replicate",
        init_quantile_offset: float = 0.05,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, feat_channels, kernel_size=3, padding_mode=conv_padding_mode)
        self.conv2 = ConvNormAct(feat_channels, feat_channels, kernel_size=3, padding_mode=conv_padding_mode)
        proj_out = num_pollutants * num_components
        self.proj = nn.Conv2d(feat_channels, proj_out, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.bias)

        if num_components == 3:
            raw0 = math.log(math.expm1(max(1e-4, float(init_quantile_offset))))
            with torch.no_grad():
                bias = self.proj.bias.view(num_pollutants, num_components)
                bias[:, 1] = raw0
                bias[:, 2] = raw0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.proj(x)


class LocalResidualHead(nn.Module):
    """6-head independent decoder with shared FiLM time conditioning.

    Each horizon has its own decoder (independent weights) that receives
    FiLM-modulated feat + per-horizon wind. Structural diversity is
    guaranteed by independent random init.
    """

    def __init__(
        self,
        feat_channels: int = 72,
        num_pollutants: int = 6,
        num_horizons: int = 6,
        wind_channels: int = 2,
        time_context_dim: int = 512,
        horizon_hours: tuple[int, ...] | None = None,
        conv_padding_mode: str = "replicate",
        quantile_levels: tuple[float, ...] | None = None,
        init_quantile_offset: float = 0.05,
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.num_pollutants = num_pollutants
        self.num_horizons = num_horizons
        self.wind_channels = wind_channels
        self.time_context_dim = int(time_context_dim)

        if quantile_levels is not None:
            levels = tuple(float(q) for q in quantile_levels)
            if len(levels) != 3 or not (0.0 < levels[0] < 0.5 < levels[2] < 1.0) or levels[1] != 0.5:
                raise ValueError(f"quantile_levels must be (lo, 0.5, hi), got {levels}")
            self.quantile_levels: tuple[float, ...] | None = levels
            self.num_components = 3
        else:
            self.quantile_levels = None
            self.num_components = 1

        self.film = HorizonFiLM(feat_channels, time_context_dim)

        in_channels = feat_channels + wind_channels
        self.heads = nn.ModuleList([
            HorizonHead(
                in_channels=in_channels,
                feat_channels=feat_channels,
                num_pollutants=num_pollutants,
                num_components=self.num_components,
                conv_padding_mode=conv_padding_mode,
                init_quantile_offset=init_quantile_offset,
            )
            for _ in range(num_horizons)
        ])

        if horizon_hours is None:
            horizon_hours = (1, 2, 4, 8, 12, 24) if num_horizons == 6 else tuple(range(1, num_horizons + 1))
        if len(horizon_hours) != num_horizons:
            raise ValueError(f"horizon_hours length mismatch: {len(horizon_hours)} vs {num_horizons}")
        self.register_buffer("_horizon_hours", torch.tensor(horizon_hours, dtype=torch.float32), persistent=False)

        self.out_channels = num_horizons * num_pollutants

    def _lead_time_encoding(self, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
        h = self._horizon_hours.to(device=device, dtype=dtype)
        return torch.log1p(h).view(1, self.num_horizons, 1).expand(batch_size, self.num_horizons, 1)

    def forward(
        self,
        feat: torch.Tensor,
        wind_uv_h2: torch.Tensor | None,
        *,
        reshape_output: bool = False,
        time_context_horizon: torch.Tensor | None = None,
        return_quantiles: bool = False,
        c0_inr: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if feat.dim() != 4:
            raise ValueError(f"feat must be [B,C,H,W], got {tuple(feat.shape)}")
        b, _c, h, w = feat.shape
        hh = self.num_horizons
        p = self.num_pollutants
        nc = self.num_components

        if wind_uv_h2 is None:
            wind_h = torch.zeros(b, hh, self.wind_channels, h, w, device=feat.device, dtype=feat.dtype)
        else:
            if wind_uv_h2.shape[1] < hh:
                raise ValueError(f"wind_uv_h2 horizons {wind_uv_h2.shape[1]} < {hh}")
            wind_h = wind_uv_h2[:, :hh].to(dtype=feat.dtype)
            if wind_h.shape[-2:] != (h, w):
                wind_h = F.interpolate(
                    wind_h.reshape(b * hh, self.wind_channels, wind_h.shape[-2], wind_h.shape[-1]),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).view(b, hh, self.wind_channels, h, w)

        lead_enc = self._lead_time_encoding(feat.device, feat.dtype, b)

        if time_context_horizon is not None:
            if time_context_horizon.shape != (b, hh, self.time_context_dim):
                raise ValueError(
                    f"time_context_horizon must be [B,Hh,{self.time_context_dim}], "
                    f"got {tuple(time_context_horizon.shape)}"
                )
            cond_all = torch.cat([time_context_horizon.to(dtype=feat.dtype), lead_enc], dim=-1)
        else:
            zeros_ctx = torch.zeros(b, hh, self.time_context_dim, device=feat.device, dtype=feat.dtype)
            cond_all = torch.cat([zeros_ctx, lead_enc], dim=-1)

        preds = []
        for i in range(hh):
            feat_mod = self.film(cond_all[:, i], feat)
            x_i = torch.cat([feat_mod, wind_h[:, i]], dim=1)
            preds.append(self.heads[i](x_i))

        raw = torch.stack(preds, dim=1).view(b, hh, p, nc, h, w)

        if nc == 1:
            delta = raw[:, :, :, 0]
            quantiles = None
        else:
            delta = raw[:, :, :, 0]
            lo_off = F.softplus(raw[:, :, :, 1])
            hi_off = F.softplus(raw[:, :, :, 2])
            quantiles = None  # built after residual add

        if c0_inr is not None:
            base = c0_inr.unsqueeze(1).expand_as(delta)
            pred = base + delta
        else:
            pred = delta

        if nc == 3:
            lo = pred - lo_off
            hi = pred + hi_off
            quantiles = torch.stack([lo, pred, hi], dim=3)

        out = pred if reshape_output else pred.reshape(b, self.out_channels, h, w)
        if return_quantiles:
            if quantiles is None:
                quantiles = pred.unsqueeze(3)
            return out, quantiles
        return out


__all__ = ["LocalResidualHead", "HorizonFiLM", "HorizonHead"]

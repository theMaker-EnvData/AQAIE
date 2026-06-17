from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_sparse_inr import ObsSparseINREncoder, resolve_inr_encoder_config
from conv_blocks import ConvNormAct


def _resolve_gn_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


def _apply_film_2d(x: torch.Tensor, film: nn.Linear, time_context: torch.Tensor) -> torch.Tensor:
    if time_context.dim() != 2 or time_context.shape[0] != x.shape[0]:
        raise ValueError(
            "time_context must be [B,D] with matching B, "
            f"got shape={tuple(time_context.shape)} for B={x.shape[0]}"
        )
    channels = x.shape[1]
    if film.out_features != 2 * channels:
        raise ValueError(
            f"FiLM output channels mismatch: expected {2 * channels}, got {film.out_features}"
        )
    raw = film(time_context)
    d_gamma, beta = torch.split(raw, [channels, channels], dim=1)
    gamma = 1.0 + d_gamma
    return x * gamma[:, :, None, None] + beta[:, :, None, None]


class PartialConv2d(nn.Module):
    """Mask-aware partial convolution for sparse observation inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

        self.input_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        mask_kernel = torch.ones(1, in_channels, kernel_size, kernel_size)
        self.register_buffer("mask_kernel", mask_kernel)
        self.slide_winsize = float(in_channels * kernel_size * kernel_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape != mask.shape:
            raise ValueError(f"partial conv input/mask shape mismatch: x={tuple(x.shape)} mask={tuple(mask.shape)}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mask = torch.clamp(mask, min=0.0, max=1.0)
        masked_x = x * mask
        raw_out = self.input_conv(masked_x)

        with torch.no_grad():
            update_mask = F.conv2d(
                mask,
                self.mask_kernel,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            valid = (update_mask > 0.0).to(dtype=masked_x.dtype)
            mask_ratio = self.slide_winsize / (update_mask + self.eps)
            mask_ratio = mask_ratio * valid

        if self.input_conv.bias is not None:
            bias = self.input_conv.bias.view(1, -1, 1, 1)
            out = (raw_out - bias) * mask_ratio + bias
            out = out * valid
        else:
            out = raw_out * mask_ratio

        return out, valid


class Temporal3DEncoderWithFiLM(nn.Module):
    """Encode temporal tensor [B,T,C,H,W] with per-lag FiLM before temporal merge."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_steps: int = 3,
        time_context_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.time_steps = int(time_steps)
        self.out_channels = int(out_channels)
        self.time_context_dim = int(time_context_dim)
        self.lag_film = nn.Linear(self.time_context_dim, 2 * in_channels)
        self.temporal_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(time_steps, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.norm = nn.GroupNorm(_resolve_gn_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, time_context_lag: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"temporal input must be [B,T,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] != self.time_steps or x.shape[2] != self.in_channels:
            raise ValueError(
                f"temporal input must be [B,{self.time_steps},{self.in_channels},H,W], got shape={tuple(x.shape)}"
            )

        if time_context_lag is not None:
            if time_context_lag.dim() != 3 or time_context_lag.shape[0] != x.shape[0]:
                raise ValueError(
                    "time_context_lag must be [B,T,D] with matching B, "
                    f"got shape={tuple(time_context_lag.shape)} for B={x.shape[0]}"
                )
            if time_context_lag.shape[1] != self.time_steps:
                raise ValueError(
                    f"time_context_lag T mismatch: expected {self.time_steps}, got {time_context_lag.shape[1]}"
                )
            if time_context_lag.shape[2] != self.time_context_dim:
                raise ValueError(
                    f"time_context_lag dim mismatch: expected {self.time_context_dim}, got {time_context_lag.shape[2]}"
                )
            modulated: list[torch.Tensor] = []
            for t in range(self.time_steps):
                feat_t = _apply_film_2d(x[:, t], self.lag_film, time_context_lag[:, t])
                modulated.append(feat_t)
            x = torch.stack(modulated, dim=1)

        # Conv3d expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.temporal_conv(x)
        x = x.squeeze(2)
        x = self.norm(x)
        x = self.act(x)
        return x


class ObsTemporalPartialConvEncoderWithFiLM(nn.Module):
    """Encode sparse obs tensor [B,T,C,H,W] with per-lag partial conv + lag FiLM + temporal merge."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_steps: int = 3,
        time_context_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_steps = int(time_steps)
        self.time_context_dim = int(time_context_dim)

        self.partial_by_time = nn.ModuleList(
            [
                PartialConv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                for _ in range(self.time_steps)
            ]
        )
        self.step_norm = nn.ModuleList(
            [nn.GroupNorm(_resolve_gn_groups(self.out_channels), self.out_channels) for _ in range(self.time_steps)]
        )
        self.step_act = nn.SiLU(inplace=True)
        self.lag_film = nn.Linear(self.time_context_dim, 2 * self.out_channels)

        self.temporal_conv = nn.Conv3d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(self.time_steps, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(_resolve_gn_groups(self.out_channels), self.out_channels)
        self.temporal_act = nn.SiLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        time_context_lag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"obs temporal input must be [B,T,C,H,W], got shape={tuple(x.shape)}")
        if valid_mask.shape != x.shape:
            raise ValueError(
                "obs temporal valid_mask shape mismatch: "
                f"x={tuple(x.shape)} mask={tuple(valid_mask.shape)}"
            )
        if x.shape[1] != self.time_steps or x.shape[2] != self.in_channels:
            raise ValueError(
                f"obs temporal input must be [B,{self.time_steps},{self.in_channels},H,W], got shape={tuple(x.shape)}"
            )
        if time_context_lag is not None:
            if time_context_lag.dim() != 3 or time_context_lag.shape[0] != x.shape[0]:
                raise ValueError(
                    "time_context_lag must be [B,T,D] with matching B, "
                    f"got shape={tuple(time_context_lag.shape)} for B={x.shape[0]}"
                )
            if time_context_lag.shape[1] != self.time_steps:
                raise ValueError(
                    f"time_context_lag T mismatch: expected {self.time_steps}, got {time_context_lag.shape[1]}"
                )
            if time_context_lag.shape[2] != self.time_context_dim:
                raise ValueError(
                    f"time_context_lag dim mismatch: expected {self.time_context_dim}, got {time_context_lag.shape[2]}"
                )

        step_feats: list[torch.Tensor] = []
        step_valid: list[torch.Tensor] = []
        for t in range(self.time_steps):
            feat_t, valid_t = self.partial_by_time[t](x[:, t], valid_mask[:, t])
            feat_t = self.step_norm[t](feat_t)
            feat_t = self.step_act(feat_t)
            if time_context_lag is not None:
                feat_t = _apply_film_2d(feat_t, self.lag_film, time_context_lag[:, t])
            feat_t = feat_t * valid_t
            step_feats.append(feat_t)
            step_valid.append(valid_t)

        stacked = torch.stack(step_feats, dim=1)
        stacked_3d = stacked.permute(0, 2, 1, 3, 4).contiguous()

        out = self.temporal_conv(stacked_3d).squeeze(2)
        out = self.temporal_norm(out)
        out = self.temporal_act(out)

        any_valid = torch.stack(step_valid, dim=1).amax(dim=1)
        out = out * any_valid
        return out


def _gaussian_blur_2d(x: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    """Apply a Gaussian blur to a [B, C, H, W] tensor using grouped Conv2d."""
    ks = int(kernel_size)
    if ks < 3 or ks % 2 == 0:
        return x
    s = max(float(sigma), 0.5)
    half = ks // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (coords / s) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, ks, ks).to(device=x.device, dtype=x.dtype)
    c = x.shape[1]
    kernel = kernel.expand(c, 1, ks, ks)
    x = F.pad(x, (half, half, half, half), mode="replicate")
    return F.conv2d(x, kernel, groups=c)


class FusionPreprocessor(nn.Module):
    """Input preprocessor that produces U-Net L0 features from obs+met only.

    Path definition:
    - dynamic = concat(obs_temporal_feat, met_temporal_feat, met_direct_21)
    - dynamic_stem: Conv3x3 x2
    - fusion_conv: Conv3x3 x1 on dynamic_stem output

    Static inputs are handled separately via StaticSpatialFiLM on decoder stages.
    """

    def __init__(
        self,
        c_tm: int = 32,
        c_obs: int = 24,
        c_mask: int = 16,
        l0_channels: int = 72,
        dynamic_stem_channels: int = 72,
        time_context_dim: int = 512,
        conv_padding_mode: str = "replicate",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.c_tm = c_tm
        self.c_obs = c_obs
        self.c_mask = c_mask
        self.l0_channels = l0_channels
        inr_cfg = resolve_inr_encoder_config(config)
        self.static_ctx_channels = int(inr_cfg["static_ctx_channels"])

        dynamic_in = c_obs + c_tm + 21

        self.obs_temporal_encoder = ObsSparseINREncoder(config=config)
        self.met_temporal_encoder = Temporal3DEncoderWithFiLM(
            in_channels=10,
            out_channels=c_tm,
            time_steps=3,
            time_context_dim=time_context_dim,
        )

        self.dynamic_stem = nn.Sequential(
            ConvNormAct(dynamic_in, dynamic_stem_channels, kernel_size=3, padding_mode=conv_padding_mode),
            ConvNormAct(dynamic_stem_channels, dynamic_stem_channels, kernel_size=3, padding_mode=conv_padding_mode),
        )
        self.fusion_conv = ConvNormAct(dynamic_stem_channels, l0_channels, kernel_size=3, padding_mode=conv_padding_mode)

        mpf = {}
        if isinstance(config, Mapping):
            backbone_cfg = config.get("backbone", {})
            if isinstance(backbone_cfg, Mapping):
                mpf = backbone_cfg.get("met_pre_filter", {})
        if not isinstance(mpf, Mapping):
            mpf = {}
        self._mpf_enabled = bool(mpf.get("enabled", False))
        self._mpf_sigma = float(mpf.get("sigma", 1.0))
        self._mpf_kernel = int(mpf.get("kernel_size", 5))
        if self._mpf_kernel % 2 == 0:
            self._mpf_kernel += 1

    def _resolve_static_ctx(self, static_16: torch.Tensor | None) -> torch.Tensor | None:
        if static_16 is None:
            return None
        if static_16.shape[1] < self.static_ctx_channels:
            raise ValueError(
                f"static_16 must have at least {self.static_ctx_channels} channels, "
                f"got {static_16.shape[1]}"
            )
        return static_16[:, : self.static_ctx_channels]

    def _apply_met_pre_filter(self, met_direct_21: torch.Tensor) -> torch.Tensor:
        if not self._mpf_enabled:
            return met_direct_21
        return _gaussian_blur_2d(met_direct_21, sigma=self._mpf_sigma, kernel_size=self._mpf_kernel)

    def forward(
        self,
        obs_stations_coord: torch.Tensor,
        obs_stations_mask: torch.Tensor,
        obs_stations_values: torch.Tensor,
        obs_stations_valid: torch.Tensor,
        met_temporal_10x3: torch.Tensor,
        met_direct_21: torch.Tensor,
        time_context_lag: torch.Tensor | None = None,
        static_16: torch.Tensor | None = None,
        return_field_lags: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if met_direct_21.dim() != 4 or met_direct_21.shape[1] != 21:
            raise ValueError(f"met_direct_21 must be [B,21,H,W], got shape={tuple(met_direct_21.shape)}")

        target_hw = met_direct_21.shape[-2:]
        if met_temporal_10x3.shape[-2:] != target_hw:
            raise ValueError(
                "met_temporal_10x3 resolution must match met_direct_21. "
                f"met={tuple(met_temporal_10x3.shape[-2:])}, target={tuple(target_hw)}"
            )

        static_ctx = self._resolve_static_ctx(static_16)
        enc_out = self.obs_temporal_encoder(
            coord=obs_stations_coord,
            station_mask=obs_stations_mask,
            values=obs_stations_values,
            valid=obs_stations_valid,
            met_temporal_10x3=met_temporal_10x3,
            met_direct_21=met_direct_21,
            time_context_lag=time_context_lag,
            static_ctx=static_ctx,
            return_field_lags=return_field_lags,
        )
        if return_field_lags:
            obs_temporal_feat, c0_inr, field_lags = enc_out
        else:
            obs_temporal_feat, c0_inr = enc_out
            field_lags = None
        met_temporal_feat = self.met_temporal_encoder(met_temporal_10x3, time_context_lag=time_context_lag)

        if met_temporal_feat.shape[-2:] != target_hw:
            raise ValueError(
                "Temporal encoder output resolution must match met_direct_21. "
                f"temporal={tuple(met_temporal_feat.shape[-2:])}, target={tuple(target_hw)}"
            )

        met_direct_21 = self._apply_met_pre_filter(met_direct_21)

        dynamic = torch.cat([obs_temporal_feat, met_temporal_feat, met_direct_21], dim=1)
        dyn_out = self.dynamic_stem(dynamic)
        l0 = self.fusion_conv(dyn_out)
        if return_field_lags:
            return l0, c0_inr, field_lags
        return l0, c0_inr

    def forward_inr_encoder_only(
        self,
        obs_stations_coord: torch.Tensor,
        obs_stations_mask: torch.Tensor,
        obs_stations_values: torch.Tensor,
        obs_stations_valid: torch.Tensor,
        met_temporal_10x3: torch.Tensor,
        met_direct_21: torch.Tensor,
        time_context_lag: torch.Tensor | None = None,
        static_16: torch.Tensor | None = None,
        precomputed_geometry: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """INR encoder + field heads only (skip met temporal stem and fusion conv)."""
        static_ctx = self._resolve_static_ctx(static_16)
        enc_out = self.obs_temporal_encoder(
            coord=obs_stations_coord,
            station_mask=obs_stations_mask,
            values=obs_stations_values,
            valid=obs_stations_valid,
            met_temporal_10x3=met_temporal_10x3,
            met_direct_21=met_direct_21,
            time_context_lag=time_context_lag,
            static_ctx=static_ctx,
            return_field_lags=True,
            precomputed_geometry=precomputed_geometry,
        )
        obs_temporal_feat, c0_inr, field_lags = enc_out
        return obs_temporal_feat, c0_inr, field_lags

    def forward_from_batch_inputs(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            obs_stations_coord=inputs["obs_stations_coord"],
            obs_stations_mask=inputs["obs_stations_mask"],
            obs_stations_values=inputs["obs_stations_values"],
            obs_stations_valid=inputs["obs_stations_valid"],
            met_temporal_10x3=inputs["met_temporal_10x3"],
            met_direct_21=inputs["met_direct_21"],
        )


__all__ = ["FusionPreprocessor", "Temporal3DEncoderWithFiLM"]

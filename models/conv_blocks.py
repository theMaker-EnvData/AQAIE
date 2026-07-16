from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

PADDING_MODES = ("zero", "replicate", "reflect")
COORD_CHANNELS = 2


def _resolve_gn_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


def normalize_padding_mode(padding_mode: str) -> str:
    mode = str(padding_mode).strip().lower()
    if mode not in PADDING_MODES:
        raise ValueError(f"conv padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}")
    return mode


def resolve_backbone_conv_options(config: Mapping[str, Any] | None) -> tuple[str, bool]:
    """Read conv padding + coordconv flags from merged config."""
    if config is None:
        return "replicate", True

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    backbone_cfg = config.get("backbone", {})
    if not isinstance(backbone_cfg, Mapping):
        backbone_cfg = {}
    if not backbone_cfg and isinstance(model_cfg.get("backbone"), Mapping):
        backbone_cfg = model_cfg["backbone"]

    padding_mode = normalize_padding_mode(str(backbone_cfg.get("conv_padding_mode", "replicate")))
    coord_cfg = backbone_cfg.get("coordconv", {})
    if not isinstance(coord_cfg, Mapping):
        coord_cfg = {}
    coordconv_enabled = bool(coord_cfg.get("enabled", True))
    return padding_mode, coordconv_enabled


def resolve_backbone_blurpool_options(config: Mapping[str, Any] | None) -> tuple[bool, int]:
    """Read encoder BlurPool flags from merged config (Zhang ICML 2019)."""
    if config is None:
        return False, 3

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    backbone_cfg = config.get("backbone", {})
    if not isinstance(backbone_cfg, Mapping):
        backbone_cfg = {}
    if not backbone_cfg and isinstance(model_cfg.get("backbone"), Mapping):
        backbone_cfg = model_cfg["backbone"]

    blur_cfg = backbone_cfg.get("blurpool", {})
    if isinstance(blur_cfg, bool):
        return bool(blur_cfg), 3
    if not isinstance(blur_cfg, Mapping):
        blur_cfg = {}

    enabled = bool(blur_cfg.get("enabled", False))
    filt_size = int(blur_cfg.get("filt_size", 3))
    if filt_size not in (3, 5):
        raise ValueError(
            "backbone.blurpool.filt_size must be 3 (binomial-3) or 5 (binomial-5), "
            f"got {filt_size}"
        )
    return enabled, filt_size


def resolve_backbone_activation_checkpoint(config: Mapping[str, Any] | None) -> bool:
    """Read U-Net activation checkpointing flag (train-time memory vs compute)."""
    if config is None:
        return False

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    backbone_cfg = config.get("backbone", {})
    if not isinstance(backbone_cfg, Mapping):
        backbone_cfg = {}
    if not backbone_cfg and isinstance(model_cfg.get("backbone"), Mapping):
        backbone_cfg = model_cfg["backbone"]

    ac_cfg = backbone_cfg.get("activation_checkpoint", False)
    if isinstance(ac_cfg, bool):
        return ac_cfg
    if isinstance(ac_cfg, Mapping):
        return bool(ac_cfg.get("enabled", False))
    return bool(ac_cfg)


def resolve_backbone_skip_blur_options(
    config: Mapping[str, Any] | None,
) -> tuple[bool, int, list[str]]:
    """Read encoder skip-connection anti-alias blur flags.

    Returns (enabled, filt_size, levels).  levels are encoder skip keys
    (e.g. ["enc0", "enc1"]) whose features are blurred before concat with
    the decoder upsampled tensor.  enc0→dec0, enc1→dec1, etc.
    """
    if config is None:
        return False, 3, []

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    backbone_cfg = config.get("backbone", {})
    if not isinstance(backbone_cfg, Mapping):
        backbone_cfg = {}
    if not backbone_cfg and isinstance(model_cfg.get("backbone"), Mapping):
        backbone_cfg = model_cfg["backbone"]

    sb_cfg = backbone_cfg.get("skip_blur", {})
    if isinstance(sb_cfg, bool):
        sb_cfg = {}
    if not isinstance(sb_cfg, Mapping):
        sb_cfg = {}

    enabled = bool(sb_cfg.get("enabled", False))
    filt_size = int(sb_cfg.get("filt_size", 3))
    if filt_size not in (3, 5):
        raise ValueError(
            "backbone.skip_blur.filt_size must be 3 (binomial-3) or 5 (binomial-5), "
            f"got {filt_size}"
        )
    levels_raw = sb_cfg.get("levels", [])
    if isinstance(levels_raw, (list, tuple)):
        levels = [str(v) for v in levels_raw]
    else:
        levels = [str(levels_raw)]
    valid = {f"enc{i}" for i in range(5)}
    for lvl in levels:
        if lvl not in valid:
            raise ValueError(
                f"backbone.skip_blur.levels: unknown level {lvl!r}; "
                f"must be in {sorted(valid)}"
            )
    return enabled, filt_size, levels


def build_coord_grid(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized grid coords in [-1, 1], align_corners=True; returns [B, 2, H, W] (y, x)."""
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(batch_size, COORD_CHANNELS, height, width)
    return grid.contiguous()


def append_coord_channels(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"append_coord_channels expects [B,C,H,W], got {tuple(x.shape)}")
    b, _, h, w = x.shape
    coord = build_coord_grid(b, h, w, x.device, x.dtype)
    return torch.cat([x, coord], dim=1)


class PaddedConv2d(nn.Module):
    """Conv2d with explicit edge padding (replicate/reflect) instead of implicit zero pad."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding_mode = normalize_padding_mode(padding_mode)
        self.edge_pad = max(0, self.kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1 or self.edge_pad == 0:
            return self.conv(x)
        if self.padding_mode == "zero":
            return F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.stride,
                padding=self.edge_pad,
            )
        x = F.pad(x, (self.edge_pad, self.edge_pad, self.edge_pad, self.edge_pad), mode=self.padding_mode)
        return self.conv(x)


class BinomialBlurPool2d(nn.Module):
    """Depthwise binomial low-pass filter + stride (BlurPool, Zhang ICML 2019).

    filt_size=3 -> binomial-3 [1,2,1]; filt_size=5 -> binomial-5 [1,4,6,4,1].
    """

    _ALLOWED_FILT_SIZES = (3, 5)

    def __init__(
        self,
        channels: int,
        filt_size: int = 3,
        stride: int = 2,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        filt_size = int(filt_size)
        if filt_size not in self._ALLOWED_FILT_SIZES:
            raise ValueError(
                f"BinomialBlurPool2d filt_size must be one of {self._ALLOWED_FILT_SIZES}, got {filt_size}"
            )
        self.channels = int(channels)
        self.filt_size = filt_size
        self.stride = int(stride)
        self.padding_mode = normalize_padding_mode(padding_mode)
        half = (self.filt_size - 1) // 2
        rem = self.filt_size - 1 - half
        self._pad = (half, rem, half, rem)

        coeffs = torch.tensor(
            [math.comb(self.filt_size - 1, k) for k in range(self.filt_size)],
            dtype=torch.float32,
        )
        coeffs = coeffs / coeffs.sum()
        kernel_2d = coeffs[:, None] * coeffs[None, :]
        filt = kernel_2d.view(1, 1, self.filt_size, self.filt_size).repeat(self.channels, 1, 1, 1)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"BinomialBlurPool2d channel mismatch: expected {self.channels}, got {x.shape[1]}"
            )
        x = F.pad(x, self._pad, mode=self.padding_mode)
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.conv = PaddedConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding_mode=padding_mode,
        )
        self.norm = nn.GroupNorm(_resolve_gn_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


__all__ = [
    "PADDING_MODES",
    "COORD_CHANNELS",
    "PaddedConv2d",
    "BinomialBlurPool2d",
    "ConvNormAct",
    "build_coord_grid",
    "append_coord_channels",
    "normalize_padding_mode",
    "resolve_backbone_conv_options",
    "resolve_backbone_blurpool_options",
    "resolve_backbone_activation_checkpoint",
    "resolve_backbone_skip_blur_options",
]
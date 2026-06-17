from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from conv_blocks import BinomialBlurPool2d, PaddedConv2d, append_coord_channels, normalize_padding_mode


def _resolve_gn_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


def _forward_double_conv(
    block: "DoubleConvWithFiLM",
    x: torch.Tensor,
    film: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    return block(x, film=film)


def _forward_downsample(block: "DownsampleBlock", x: torch.Tensor) -> torch.Tensor:
    return block(x)


def _forward_upsample(
    block: "UpsampleBlock",
    x: torch.Tensor,
    skip: torch.Tensor,
    film: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    return block(x, skip, film=film)


class DoubleConvWithFiLM(nn.Module):
    """Two Conv3x3 blocks where FiLM is applied after conv2 normalization."""

    def __init__(self, in_channels: int, out_channels: int, padding_mode: str = "replicate") -> None:
        super().__init__()
        self.padding_mode = normalize_padding_mode(padding_mode)
        self.conv1 = PaddedConv2d(in_channels, out_channels, kernel_size=3, padding_mode=self.padding_mode)
        self.norm1 = nn.GroupNorm(_resolve_gn_groups(out_channels), out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = PaddedConv2d(out_channels, out_channels, kernel_size=3, padding_mode=self.padding_mode)
        self.norm2 = nn.GroupNorm(_resolve_gn_groups(out_channels), out_channels)
        self.act2 = nn.SiLU(inplace=True)

    @staticmethod
    def _reshape_film_param(v: torch.Tensor, channels: int) -> torch.Tensor:
        if v.dim() == 2:
            return v.unsqueeze(-1).unsqueeze(-1)
        if v.dim() == 4:
            return v
        raise ValueError(f"FiLM parameter must be [B,C] or [B,C,1,1], got shape={tuple(v.shape)}")

    def _apply_film(
        self,
        x: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if film is None:
            return x
        gamma, beta = film
        gamma = self._reshape_film_param(gamma, x.shape[1])
        beta = self._reshape_film_param(beta, x.shape[1])
        if gamma.shape[1] != x.shape[1] or beta.shape[1] != x.shape[1]:
            raise ValueError(
                "FiLM channel mismatch: "
                f"x={x.shape[1]}, gamma={gamma.shape[1]}, beta={beta.shape[1]}"
            )
        return gamma * x + beta

    def forward(
        self,
        x: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self._apply_film(x, film)
        x = self.act2(x)
        return x


class DownsampleBlock(nn.Module):
    """Stride-2 downsample; optional binomial BlurPool before 3x3 conv (anti-alias)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
        blurpool: bool = False,
        blurpool_filt_size: int = 3,
    ) -> None:
        super().__init__()
        pad = normalize_padding_mode(padding_mode)
        if blurpool:
            self.blur = BinomialBlurPool2d(
                in_channels,
                filt_size=blurpool_filt_size,
                stride=2,
                padding_mode=pad,
            )
            self.down = PaddedConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding_mode=pad,
            )
        else:
            self.blur = None
            self.down = PaddedConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding_mode=pad,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blur is not None:
            x = self.blur(x)
        return self.down(x)


class UpsampleBlock(nn.Module):
    """Bilinear upsample + 1x1 projection followed by skip concat and DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        padding_mode: str = "replicate",
        skip_blur: Optional[BinomialBlurPool2d] = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = DoubleConvWithFiLM(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            padding_mode=padding_mode,
        )
        self.skip_blur = skip_blur

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.proj(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        if self.skip_blur is not None:
            skip = self.skip_blur(skip)

        x = torch.cat([x, skip], dim=1)
        return self.block(x, film=film)


class BackboneUNet(nn.Module):
    """5-stage U-Net backbone.

    Interface contract:
    - Input is fusion_conv output at L0 width (optionally + coord channels at enc0).
    - Output is decoder L0 feature map for downstream head.
    - FiLM is injected after the second conv normalization of each stage.
    - Optional activation checkpointing on enc/dec blocks (train-only memory trade).
    """

    def __init__(
        self,
        in_channels: int = 72,
        widths: Tuple[int, int, int, int, int, int] = (72, 144, 288, 432, 576, 768),
        padding_mode: str = "replicate",
        coordconv_enabled: bool = True,
        blurpool: bool = False,
        blurpool_filt_size: int = 3,
        activation_checkpoint: bool = False,
        skip_blur_enabled: bool = False,
        skip_blur_filt_size: int = 3,
        skip_blur_levels: list[str] | None = None,
    ) -> None:
        super().__init__()
        l0, l1, l2, l3, l4, bottleneck = widths
        self.l0_channels = int(l0)
        self.padding_mode = normalize_padding_mode(padding_mode)
        self.coordconv_enabled = bool(coordconv_enabled)
        self.activation_checkpoint = bool(activation_checkpoint)
        enc0_in = l0 + (2 if self.coordconv_enabled else 0)

        if in_channels != l0:
            raise ValueError(f"Backbone input channels must match L0 width. got in_channels={in_channels}, l0={l0}")

        pad = self.padding_mode
        down_kw = dict(padding_mode=pad, blurpool=blurpool, blurpool_filt_size=blurpool_filt_size)

        self.enc0 = DoubleConvWithFiLM(enc0_in, l0, padding_mode=pad)
        self.down01 = DownsampleBlock(l0, l1, **down_kw)

        self.enc1 = DoubleConvWithFiLM(l1, l1, padding_mode=pad)
        self.down12 = DownsampleBlock(l1, l2, **down_kw)

        self.enc2 = DoubleConvWithFiLM(l2, l2, padding_mode=pad)
        self.down23 = DownsampleBlock(l2, l3, **down_kw)

        self.enc3 = DoubleConvWithFiLM(l3, l3, padding_mode=pad)
        self.down34 = DownsampleBlock(l3, l4, **down_kw)

        self.enc4 = DoubleConvWithFiLM(l4, l4, padding_mode=pad)
        self.down4b = DownsampleBlock(l4, bottleneck, **down_kw)

        self.bottleneck = DoubleConvWithFiLM(bottleneck, bottleneck, padding_mode=pad)

        skip_blur_enabled = bool(skip_blur_enabled)
        skip_blur_levels_set = set(skip_blur_levels or [])
        _skip_ch = {"enc0": l0, "enc1": l1, "enc2": l2, "enc3": l3, "enc4": l4}
        _sb = {}
        if skip_blur_enabled:
            for key, ch in _skip_ch.items():
                if key in skip_blur_levels_set:
                    _sb[key] = BinomialBlurPool2d(
                        ch, filt_size=skip_blur_filt_size, stride=1, padding_mode=pad
                    )

        self.dec4 = UpsampleBlock(bottleneck, l4, l4, padding_mode=pad, skip_blur=_sb.get("enc4"))
        self.dec3 = UpsampleBlock(l4, l3, l3, padding_mode=pad, skip_blur=_sb.get("enc3"))
        self.dec2 = UpsampleBlock(l3, l2, l2, padding_mode=pad, skip_blur=_sb.get("enc2"))
        self.dec1 = UpsampleBlock(l2, l1, l1, padding_mode=pad, skip_blur=_sb.get("enc1"))
        self.dec0 = UpsampleBlock(l1, l0, l0, padding_mode=pad, skip_blur=_sb.get("enc0"))

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.coordconv_enabled:
            return x
        return append_coord_channels(x)

    def _should_checkpoint(self, *tensors: Any) -> bool:
        if not self.activation_checkpoint or not self.training:
            return False
        if not torch.is_grad_enabled():
            return False
        return any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)

    def _run_checkpointed(
        self,
        fn: Callable[..., torch.Tensor],
        *args: Any,
    ) -> torch.Tensor:
        if not self._should_checkpoint(*args):
            return fn(*args)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

    def run_double_conv(
        self,
        block: DoubleConvWithFiLM,
        x: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        return self._run_checkpointed(_forward_double_conv, block, x, film)

    def run_downsample(self, block: DownsampleBlock, x: torch.Tensor) -> torch.Tensor:
        return self._run_checkpointed(_forward_downsample, block, x)

    def run_upsample(
        self,
        block: UpsampleBlock,
        x: torch.Tensor,
        skip: torch.Tensor,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        return self._run_checkpointed(_forward_upsample, block, x, skip, film)

    @staticmethod
    def _parse_film_pair(value: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if value is None:
            return None
        if isinstance(value, tuple) and len(value) == 2:
            return value
        if isinstance(value, Mapping):
            gamma = value.get("gamma")
            beta = value.get("beta")
            if gamma is not None and beta is not None:
                return gamma, beta
        raise ValueError("FiLM entry must be (gamma,beta) tuple or mapping with gamma/beta")

    def _get_film(
        self,
        film_params: Optional[Mapping[str, Any]],
        key: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not film_params:
            return None
        return self._parse_film_pair(film_params.get(key))

    def forward_encoder_bottleneck(
        self,
        x: torch.Tensor,
        film_params: Optional[Mapping[str, Any]] = None,
        *,
        preprocess: bool = True,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if preprocess:
            x = self.preprocess_input(x)

        e0 = self.run_double_conv(self.enc0, x, self._get_film(film_params, "enc0"))
        x1 = self.run_downsample(self.down01, e0)

        e1 = self.run_double_conv(self.enc1, x1, self._get_film(film_params, "enc1"))
        x2 = self.run_downsample(self.down12, e1)

        e2 = self.run_double_conv(self.enc2, x2, self._get_film(film_params, "enc2"))
        x3 = self.run_downsample(self.down23, e2)

        e3 = self.run_double_conv(self.enc3, x3, self._get_film(film_params, "enc3"))
        x4 = self.run_downsample(self.down34, e3)

        e4 = self.run_double_conv(self.enc4, x4, self._get_film(film_params, "enc4"))
        xb = self.run_downsample(self.down4b, e4)

        b = self.run_double_conv(self.bottleneck, xb, self._get_film(film_params, "bottleneck"))

        skips: Mapping[str, torch.Tensor] = {
            "enc0": e0,
            "enc1": e1,
            "enc2": e2,
            "enc3": e3,
            "enc4": e4,
        }
        return b, skips

    def forward_decoder(
        self,
        b: torch.Tensor,
        skips: Mapping[str, torch.Tensor],
        film_params: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        d4 = self.run_upsample(self.dec4, b, skips["enc4"], self._get_film(film_params, "dec4"))
        d3 = self.run_upsample(self.dec3, d4, skips["enc3"], self._get_film(film_params, "dec3"))
        d2 = self.run_upsample(self.dec2, d3, skips["enc2"], self._get_film(film_params, "dec2"))
        d1 = self.run_upsample(self.dec1, d2, skips["enc1"], self._get_film(film_params, "dec1"))
        d0 = self.run_upsample(self.dec0, d1, skips["enc0"], self._get_film(film_params, "dec0"))
        return d0

    def forward_stages(
        self,
        x: torch.Tensor,
        film_params: Optional[Mapping[str, Any]] = None,
        *,
        preprocess: bool = True,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        b, skips = self.forward_encoder_bottleneck(x, film_params, preprocess=preprocess)
        d0 = self.forward_decoder(b, skips, film_params)
        feats: Mapping[str, torch.Tensor] = dict(skips)
        feats["bottleneck"] = b
        feats["dec0"] = d0
        return d0, feats

    def forward(
        self,
        x: torch.Tensor,
        film_params: Optional[Mapping[str, Any]] = None,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        d0, feats = self.forward_stages(x, film_params, preprocess=True)
        if not return_features:
            return d0
        return d0, feats


__all__ = ["BackboneUNet"]

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_gn_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(_resolve_gn_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class BranchCamsEncoder(nn.Module):
    """CAMS(+GEMS) branch encoder + projection and decoder gates.

    Input: 26 channels = cams_era5_20 + support_frac_1 + gems_4 + daylight_1
    Encoder: 26->96->192->192->256
    Projections:
    - species/feature map: 256->256
    - bottleneck map: 256->768
    - mid-decoder (D2) map: 256->C_D2
    Gates:
    - decoder gate vector: 256 -> sum(decoder widths), sigmoid constrained
    - gate bias is initialized to negative value for small initial influence
    """

    def __init__(
        self,
        input_channels: int = 26,
        stem_channels: int = 96,
        block_channels: tuple[int, int, int] = (192, 192, 256),
        bottleneck_channels: int = 768,
        mid_decoder_d2_channels: int = 288,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        b1, b2, b3 = block_channels

        # Encoder path
        self.stem = ConvNormAct(input_channels, stem_channels, kernel_size=3)
        self.block1 = ConvNormAct(stem_channels, b1, kernel_size=3)
        self.block2 = ConvNormAct(b1, b2, kernel_size=3)
        self.block3 = ConvNormAct(b2, b3, kernel_size=3)

        # Projection heads
        self.species_projection = nn.Conv2d(b3, 256, kernel_size=1, bias=False)
        self.bottleneck_projection = nn.Conv2d(256, bottleneck_channels, kernel_size=1, bias=False)
        self.mid_decoder_d2_projection = nn.Conv2d(256, mid_decoder_d2_channels, kernel_size=1, bias=False)

        # Gate heads for two injection points: bottleneck and decoder D2.
        self.bottleneck_gate_head = nn.Linear(256, bottleneck_channels, bias=True)
        self.d2_gate_head = nn.Linear(256, mid_decoder_d2_channels, bias=True)
        nn.init.constant_(self.bottleneck_gate_head.bias, gate_bias_init)
        nn.init.constant_(self.d2_gate_head.bias, gate_bias_init)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 26:
            raise ValueError(f"branch input must be [B,26,H,W], got shape={tuple(x.shape)}")
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def _compute_injection_gates(self, species_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(species_map, output_size=1).flatten(1)
        gate_bottleneck = torch.sigmoid(self.bottleneck_gate_head(pooled)).unsqueeze(-1).unsqueeze(-1)
        gate_d2 = torch.sigmoid(self.d2_gate_head(pooled)).unsqueeze(-1).unsqueeze(-1)
        return {
            "gate_bottleneck": gate_bottleneck,
            "gate_d2": gate_d2,
        }

    def forward(self, branch_input_26: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        z_branch_map = self.encode(branch_input_26)
        z_species_map = self.species_projection(z_branch_map)
        bottleneck_map = self.bottleneck_projection(z_species_map)
        d2_map = self.mid_decoder_d2_projection(z_species_map)
        injection_gates = self._compute_injection_gates(z_species_map)

        return {
            "z_branch_map": z_branch_map,
            "z_species_map": z_species_map,
            "bottleneck_map": bottleneck_map,
            "d2_map": d2_map,
            "injection_gates": injection_gates,
        }

    def forward_from_batch_inputs(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        cams_20 = inputs.get("branch_cams_era5_20")
        support_1 = inputs.get("branch_support_frac_1")
        gems_4 = inputs.get("branch_gems_4")
        daylight_1 = inputs.get("branch_daylight_1")

        if cams_20 is None or support_1 is None or gems_4 is None or daylight_1 is None:
            raise ValueError("Branch inputs are required: cams_20, support_1, gems_4, daylight_1")

        target_hw = cams_20.shape[-2:]
        if support_1.shape[-2:] != target_hw:
            support_1 = F.interpolate(support_1, size=target_hw, mode="bilinear", align_corners=False)
        if gems_4.shape[-2:] != target_hw:
            gems_4 = F.interpolate(gems_4, size=target_hw, mode="bilinear", align_corners=False)
        if daylight_1.shape[-2:] != target_hw:
            daylight_1 = F.interpolate(daylight_1, size=target_hw, mode="bilinear", align_corners=False)

        branch_input_26 = torch.cat([cams_20, support_1, gems_4, daylight_1], dim=1)
        return self.forward(branch_input_26)


__all__ = ["BranchCamsEncoder"]

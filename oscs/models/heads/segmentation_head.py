"""Prototype-mask segmentation head."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_prototypes: int) -> None:
        super().__init__()
        self.prototypes = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_prototypes, kernel_size=1),
        )
        self.coefficients = nn.Conv2d(in_channels, num_prototypes, kernel_size=1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        prototypes = self.prototypes(features)
        coefficients = self.coefficients(features).flatten(2).transpose(1, 2)
        instance_masks = torch.einsum("bqn,bnhw->bqhw", coefficients, prototypes)
        return coefficients, prototypes, instance_masks

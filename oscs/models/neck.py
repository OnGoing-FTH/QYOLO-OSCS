"""Small FPN/PAN neck for the YOLO-OSCS scaffold."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn

from .ultralytics_blocks import C3k2, Conv


class PANNeck(nn.Module):
    def __init__(self, channels: tuple[int, int, int], depth: int = 1) -> None:
        super().__init__()
        c3, c4, c5 = channels
        self.reduce_p5 = Conv(c5, c4, 1, 1)
        self.fuse_p4 = C3k2(c4 + c4, c4, n=depth)
        self.reduce_p4 = Conv(c4, c3, 1, 1)
        self.fuse_p3 = C3k2(c3 + c3, c3, n=depth)
        self.down_p3 = Conv(c3, c4, 3, 2)
        self.out_p4 = C3k2(c4 + c4, c4, n=depth)
        self.down_p4 = Conv(c4, c5, 3, 2)
        self.out_p5 = C3k2(c5 + c5, c5, n=depth)
        self.channels = (c3, c4, c5)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = features
        p5_reduced = self.reduce_p5(p5)
        p4_fused = self.fuse_p4(
            nn_concat(p4, functional.interpolate(p5_reduced, size=p4.shape[-2:], mode="nearest"))
        )
        p4_reduced = self.reduce_p4(p4_fused)
        p3_out = self.fuse_p3(nn_concat(p3, functional.interpolate(p4_reduced, size=p3.shape[-2:], mode="nearest")))
        p4_out = self.out_p4(nn_concat(self.down_p3(p3_out), p4_fused))
        p5_out = self.out_p5(nn_concat(self.down_p4(p4_out), p5))
        return p3_out, p4_out, p5_out


def nn_concat(left: Tensor, right: Tensor) -> Tensor:
    import torch

    return torch.cat((left, right), dim=1)

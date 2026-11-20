"""YOLOv11-inspired feature extractor built from Ultralytics blocks."""

from __future__ import annotations

from torch import Tensor, nn

from .ultralytics_blocks import C2PSA, C3k2, Conv, SPPF


class YOLOv11Backbone(nn.Module):
    """Returns P3, P4, and P5 feature maps at strides 8, 16, and 32."""

    def __init__(self, in_channels: int = 3, width: int = 32, depth: int = 1) -> None:
        super().__init__()
        c2, c3, c4, c5 = width, width * 2, width * 4, width * 8
        self.stem = Conv(in_channels, c2, 3, 2)
        self.stage2 = nn.Sequential(Conv(c2, c3, 3, 2), C3k2(c3, c3, n=depth))
        self.stage3 = nn.Sequential(Conv(c3, c4, 3, 2), C3k2(c4, c4, n=depth))
        self.stage4 = nn.Sequential(Conv(c4, c5, 3, 2), C3k2(c5, c5, n=depth))
        self.stage5 = nn.Sequential(Conv(c5, c5, 3, 2), C3k2(c5, c5, n=depth), SPPF(c5, c5), C2PSA(c5, c5, n=depth))
        self.channels = (c4, c5, c5)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(images)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5

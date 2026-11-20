"""YOLOv11-inspired feature extractor built from Ultralytics blocks."""

from __future__ import annotations

from torch import Tensor, nn

from .components import PixelUnshuffleStem
from .ultralytics_blocks import C2PSA, C3k2, Conv, SPPF


class YOLOv11Backbone(nn.Module):
    """Returns P3, P4, and P5 feature maps at strides 8, 16, and 32."""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        depth: int = 1,
        use_pixel_unshuffle_stem: bool = True,
        pixel_unshuffle_factor: int = 4,
    ) -> None:
        super().__init__()
        c2, c3, c4, c5 = width, width * 2, width * 4, width * 8
        self.use_pixel_unshuffle_stem = use_pixel_unshuffle_stem
        if use_pixel_unshuffle_stem:
            self.stem = PixelUnshuffleStem(
                in_channels=in_channels,
                hidden_channels=c2 * 2,
                downscale_factor=pixel_unshuffle_factor,
            )
            stem_channels = self.stem.out_channels
        else:
            self.stem = Conv(in_channels, c2, 3, 2)
            stem_channels = c2
        self.stage2 = nn.Sequential(Conv(stem_channels, c3, 3, 2), C3k2(c3, c3, n=depth))
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


if __name__ == "__main__":
    import torch

    backbone = YOLOv11Backbone(width=16)
    p3, p4, p5 = backbone(torch.randn(1, 3, 128, 128))
    print({"p3": tuple(p3.shape), "p4": tuple(p4.shape), "p5": tuple(p5.shape)})

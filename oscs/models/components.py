"""Reusable model components kept independent from backbone topology."""

from __future__ import annotations

from torch import Tensor, nn


class PixelUnshuffleStem(nn.Module):
    """Spatial-to-channel stem: PixelUnshuffle(r) -> 1x1 Conv(C) -> Norm -> 3x3 Conv(C/2).

    For an input ``[B, Cin, H, W]``, the feature flow is:

    - ``PixelUnshuffle(r)``: ``[B, Cin * r^2, H / r, W / r]``.
    - ``1x1 Conv``: ``[B, C, H / r, W / r]`` for channel projection.
    - ``Norm``: keeps ``[B, C, H / r, W / r]``.
    - ``3x3 Conv``: ``[B, C / 2, H / r, W / r]`` for local refinement.

    With ``Cin=3``, ``H=W=128``, ``r=4``, and ``C=64``, the tensors are
    ``[B, 3, 128, 128] -> [B, 48, 32, 32] -> [B, 64, 32, 32] ->
    [B, 64, 32, 32] -> [B, 32, 32, 32]``. The input height and width must
    both be divisible by ``downscale_factor``.
    """

    def __init__(
        self,
        in_channels: int,
        downscale_factor: int = 4,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if downscale_factor < 1:
            raise ValueError("downscale_factor must be at least 1.")

        expanded_channels = in_channels * downscale_factor**2
        self.downscale_factor = downscale_factor
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.project = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(expanded_channels)
        self.activation = nn.GELU()
        self.refine = nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=3, padding=1, bias=False)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_features(images)["output"]

    def forward_with_features(self, images: Tensor) -> dict[str, Tensor]:
        """Run the stem and expose each tensor for architecture inspection."""
        pixel_unshuffled = self.pixel_unshuffle(images)
        projected = self.project(pixel_unshuffled)
        normalized = self.norm(projected)
        activated = self.activation(normalized)
        output = self.refine(activated)
        return {
            "input": images,
            "pixel_unshuffle": pixel_unshuffled,
            "project": projected,
            "norm": normalized,
            "output": output,
        }


class ChannelHalvingBlock(nn.Module):
    """Refine C-channel features and project them to C/2 channels.

    Feature flow: ``3x3 Conv(C) -> Norm(C) -> GELU -> 1x1 Conv(C/2)``.
    The normalization and activation follow the spatial convolution, where
    feature distributions are transformed most strongly. The final ``1x1``
    convolution is a linear channel projection so downstream components can
    decide whether another normalization or activation is appropriate.
    """

    def __init__(
        self,
        in_channels: int,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if in_channels < 2 or in_channels % 2:
            raise ValueError("in_channels must be an even integer greater than or equal to 2.")

        self.in_channels = in_channels
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.norm = norm_layer(in_channels)
        self.activation = nn.GELU()
        self.channel_project = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    @property
    def out_channels(self) -> int:
        return self.in_channels // 2

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_with_features(features)["output"]

    def forward_with_features(self, features: Tensor) -> dict[str, Tensor]:
        """Run the block and expose intermediate tensors for feature inspection."""
        spatial_features = self.spatial_conv(features)
        normalized = self.norm(spatial_features)
        activated = self.activation(normalized)
        output = self.channel_project(activated)
        return {
            "input": features,
            "spatial_conv": spatial_features,
            "norm": normalized,
            "activation": activated,
            "output": output,
        }


if __name__ == "__main__":
    import torch

    stem = PixelUnshuffleStem(in_channels=3)
    stem_trace = stem.forward_with_features(torch.randn(1, 3, 128, 128))
    print({"pixel_unshuffle_stem": {name: tuple(feature.shape) for name, feature in stem_trace.items()}})

    channel_block = ChannelHalvingBlock(in_channels=48)
    channel_trace = channel_block.forward_with_features(torch.randn(1, 48, 32, 32))
    print({"channel_halving_block": {name: tuple(feature.shape) for name, feature in channel_trace.items()}})

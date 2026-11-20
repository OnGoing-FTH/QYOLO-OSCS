"""Fixed-resolution adapters and adaptive fusion for neighboring image scales.

This module never resizes inputs. The data pipeline must prepare RGB tensors at
5120, 2560, 1280, and/or 640 square pixels before they reach this module.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from oscs.utils.types import MultiResolutionOutput

from .components import ChannelHalvingBlock, PixelUnshuffleStem


SUPPORTED_RESOLUTIONS = (5120, 2560, 1280, 640)


class ResolutionAdapter(nn.Module):
    """Align one supported RGB resolution to ``[B, 3, 640, 640]``.

    Supported feature flows:

    - 5120: ``Stem(r=2) -> Stem(r=4) -> Half(12->6) -> Half(6->3)``.
    - 2560: ``Stem(r=2) -> Stem(r=2)``.
    - 1280: ``Stem(r=2)``.
    - 640: ``Identity``.

    Each input must be a square RGB tensor with its adapter's exact resolution.
    Resizing belongs to dataset loading or preprocessing, never to this module.
    """

    _PIXEL_UNSHUFFLE_FACTORS = {
        5120: (2, 4),
        2560: (2, 2),
        1280: (2,),
        640: (),
    }
    _CHANNEL_HALVING_DEPTH = {
        5120: 2,
        2560: 0,
        1280: 0,
        640: 0,
    }

    def __init__(self, input_resolution: int, in_channels: int = 3) -> None:
        super().__init__()
        if input_resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"input_resolution must be one of {SUPPORTED_RESOLUTIONS}, got {input_resolution}.")
        if in_channels != 3:
            raise ValueError("ResolutionAdapter currently accepts RGB inputs only: in_channels must equal 3.")

        self.input_resolution = input_resolution
        self.in_channels = in_channels
        factors = self._PIXEL_UNSHUFFLE_FACTORS[input_resolution]
        self.stems = nn.ModuleList(
            [PixelUnshuffleStem(in_channels=in_channels, downscale_factor=factor) for factor in factors]
        )

        channels = in_channels
        for factor in factors:
            channels = channels * factor**2 // 4
        self.channel_blocks = nn.ModuleList(
            [ChannelHalvingBlock(in_channels=channels // (2**index)) for index in range(self._CHANNEL_HALVING_DEPTH[input_resolution])]
        )
        self.out_channels = channels // (2 ** self._CHANNEL_HALVING_DEPTH[input_resolution])
        if self.out_channels != 3:
            raise RuntimeError(f"Resolution {input_resolution} does not align to three output channels.")

    def forward(self, features: Tensor) -> Tensor:
        self._validate_input(features)
        output = features
        for stem in self.stems:
            output = stem(output)
        for block in self.channel_blocks:
            output = block(output)
        return output

    def forward_with_features(self, features: Tensor) -> dict[str, Tensor]:
        """Expose named branch tensors without performing any resize operation."""
        self._validate_input(features)
        trace = {"input": features}
        output = features
        for index, stem in enumerate(self.stems, start=1):
            output = stem(output)
            trace[f"pixel_unshuffle_stem_{index}"] = output
        for index, block in enumerate(self.channel_blocks, start=1):
            output = block(output)
            trace[f"channel_halving_block_{index}"] = output
        trace["output"] = output
        return trace

    def _validate_input(self, features: Tensor) -> None:
        expected_shape = (self.input_resolution, self.input_resolution)
        if features.ndim != 4:
            raise ValueError(f"Expected a rank-4 tensor [B, 3, H, W], got shape {tuple(features.shape)}.")
        if features.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {features.shape[1]}.")
        if tuple(features.shape[-2:]) != expected_shape:
            raise ValueError(
                f"ResolutionAdapter({self.input_resolution}) expects spatial size {expected_shape}, "
                f"got {tuple(features.shape[-2:])}. Resize in the data pipeline."
            )


class AdaptiveNeighborFusion(nn.Module):
    """Learn branch weights and fuse aligned neighboring-resolution features.

    Inputs are already aligned tensors with identical ``[B, 3, 640, 640]``
    shapes. Branch weights come from global descriptors and are normalized with
    softmax. A convolutional refinement then maps the weighted RGB feature to
    ``fusion_channels`` channels.
    """

    def __init__(self, branch_count: int = 4, in_channels: int = 3, fusion_channels: int = 32) -> None:
        super().__init__()
        if branch_count < 1:
            raise ValueError("branch_count must be at least 1.")
        self.branch_count = branch_count
        self.in_channels = in_channels
        self.fusion_channels = fusion_channels
        descriptor_channels = branch_count * in_channels
        hidden_channels = max(descriptor_channels, fusion_channels // 2)
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(descriptor_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, branch_count),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.GELU(),
        )

    def forward(self, aligned_features: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        if len(aligned_features) != self.branch_count:
            raise ValueError(f"Expected {self.branch_count} aligned branches, got {len(aligned_features)}.")
        self._validate_shapes(aligned_features)
        concatenated = torch.cat(aligned_features, dim=1)
        weights = self.weight_predictor(concatenated).softmax(dim=1)
        stacked = torch.stack(aligned_features, dim=1)
        weighted = (stacked * weights[:, :, None, None, None]).sum(dim=1)
        return self.refine(weighted), weights

    def _validate_shapes(self, aligned_features: tuple[Tensor, ...]) -> None:
        reference_shape = tuple(aligned_features[0].shape)
        if len(reference_shape) != 4 or reference_shape[1] != self.in_channels:
            raise ValueError(f"Expected [B, {self.in_channels}, H, W] aligned features, got {reference_shape}.")
        if any(tuple(feature.shape) != reference_shape for feature in aligned_features[1:]):
            raise ValueError("All neighboring-resolution features must have exactly the same shape before fusion.")


class MultiResolutionNeighborAdapter(nn.Module):
    """Adapt fixed RGB resolutions to 640 and fuse available neighboring scales.

    The module accepts a mapping keyed by supported resolutions, such as
    ``{5120: x_5120, 2560: x_2560, 1280: x_1280, 640: x_640}``. It does not
    resize tensors; input generation and resize operations belong upstream.
    """

    def __init__(self, fusion_channels: int = 32) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {str(resolution): ResolutionAdapter(input_resolution=resolution) for resolution in SUPPORTED_RESOLUTIONS}
        )
        self.fusion = AdaptiveNeighborFusion(
            branch_count=len(SUPPORTED_RESOLUTIONS),
            in_channels=3,
            fusion_channels=fusion_channels,
        )

    def forward(self, inputs: Mapping[int, Tensor]) -> MultiResolutionOutput:
        missing_resolutions = [resolution for resolution in SUPPORTED_RESOLUTIONS if resolution not in inputs]
        unexpected_resolutions = sorted(set(inputs) - set(SUPPORTED_RESOLUTIONS))
        if missing_resolutions or unexpected_resolutions:
            raise ValueError(
                f"Expected exactly {SUPPORTED_RESOLUTIONS}; missing={missing_resolutions}, unexpected={unexpected_resolutions}."
            )

        aligned_features = {
            resolution: self.adapters[str(resolution)](inputs[resolution]) for resolution in SUPPORTED_RESOLUTIONS
        }
        fused, weights = self.fusion(tuple(aligned_features[resolution] for resolution in SUPPORTED_RESOLUTIONS))
        return MultiResolutionOutput(aligned_features=aligned_features, fusion_weights=weights, fused=fused)


if __name__ == "__main__":
    with torch.inference_mode():
        adapter = ResolutionAdapter(input_resolution=5120).eval()
        output = adapter(torch.randn(1, 3, 5120, 5120))
    print({"input_resolution": 5120, "output": tuple(output.shape), "out_channels": adapter.out_channels})

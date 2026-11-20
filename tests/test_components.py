import pytest
import torch

from oscs.models.components import ChannelHalvingBlock, PixelUnshuffleStem


def test_pixel_unshuffle_stem_reduces_spatial_size_and_uses_input_channels() -> None:
    component = PixelUnshuffleStem(in_channels=3, downscale_factor=4)
    output = component(torch.randn(2, 3, 128, 128))

    assert output.shape == (2, 12, 32, 32)


def test_pixel_unshuffle_stem_exposes_feature_flow() -> None:
    component = PixelUnshuffleStem(in_channels=3, downscale_factor=4)
    trace = component.forward_with_features(torch.randn(1, 3, 128, 128))

    assert {name: tuple(feature.shape) for name, feature in trace.items()} == {
        "input": (1, 3, 128, 128),
        "pixel_unshuffle": (1, 48, 32, 32),
        "project": (1, 48, 32, 32),
        "norm": (1, 48, 32, 32),
        "output": (1, 12, 32, 32),
    }


def test_channel_halving_block_requires_even_input_channels() -> None:
    with pytest.raises(ValueError, match="even"):
        ChannelHalvingBlock(in_channels=31)

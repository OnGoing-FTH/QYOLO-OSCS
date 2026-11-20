import pytest
import torch

from oscs.models.components import PixelUnshuffleStem


def test_pixel_unshuffle_stem_reduces_spatial_size_and_outputs_half_c() -> None:
    component = PixelUnshuffleStem(in_channels=3, hidden_channels=32)
    output = component(torch.randn(2, 3, 128, 128))

    assert component.out_channels == 16
    assert output.shape == (2, 16, 32, 32)


def test_pixel_unshuffle_stem_exposes_feature_flow() -> None:
    component = PixelUnshuffleStem(in_channels=3, hidden_channels=64)
    trace = component.forward_with_features(torch.randn(1, 3, 128, 128))

    assert {name: tuple(feature.shape) for name, feature in trace.items()} == {
        "input": (1, 3, 128, 128),
        "pixel_unshuffle": (1, 48, 32, 32),
        "project": (1, 64, 32, 32),
        "norm": (1, 64, 32, 32),
        "output": (1, 32, 32, 32),
    }


def test_pixel_unshuffle_stem_requires_even_hidden_channels() -> None:
    with pytest.raises(ValueError, match="even"):
        PixelUnshuffleStem(in_channels=3, hidden_channels=31)

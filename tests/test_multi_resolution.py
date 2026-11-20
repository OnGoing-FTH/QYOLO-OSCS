import torch

from oscs.models.multi_resolution import ResolutionAdapter


def test_5120_resolution_adapter_aligns_rgb_features_to_640() -> None:
    adapter = ResolutionAdapter(input_resolution=5120).eval()
    with torch.inference_mode():
        trace = adapter.forward_with_features(torch.randn(1, 3, 5120, 5120))

    assert {name: tuple(feature.shape) for name, feature in trace.items()} == {
        "input": (1, 3, 5120, 5120),
        "pixel_unshuffle_stem_1": (1, 3, 2560, 2560),
        "pixel_unshuffle_stem_2": (1, 12, 640, 640),
        "channel_halving_block_1": (1, 6, 640, 640),
        "channel_halving_block_2": (1, 3, 640, 640),
        "output": (1, 3, 640, 640),
    }

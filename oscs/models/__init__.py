"""Model components and the assembled YOLO-OSCS network."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components import ChannelHalvingBlock, PixelUnshuffleStem
    from .model import YOLOOpenSetColdStartSegmentation
    from .multi_resolution import AdaptiveNeighborFusion, MultiResolutionNeighborAdapter, ResolutionAdapter

__all__ = [
    "AdaptiveNeighborFusion",
    "ChannelHalvingBlock",
    "MultiResolutionNeighborAdapter",
    "PixelUnshuffleStem",
    "ResolutionAdapter",
    "YOLOOpenSetColdStartSegmentation",
]


def __getattr__(name: str):
    if name == "ChannelHalvingBlock":
        from .components import ChannelHalvingBlock

        return ChannelHalvingBlock
    if name == "PixelUnshuffleStem":
        from .components import PixelUnshuffleStem

        return PixelUnshuffleStem
    if name == "ResolutionAdapter":
        from .multi_resolution import ResolutionAdapter

        return ResolutionAdapter
    if name == "AdaptiveNeighborFusion":
        from .multi_resolution import AdaptiveNeighborFusion

        return AdaptiveNeighborFusion
    if name == "MultiResolutionNeighborAdapter":
        from .multi_resolution import MultiResolutionNeighborAdapter

        return MultiResolutionNeighborAdapter
    if name == "YOLOOpenSetColdStartSegmentation":
        from .model import YOLOOpenSetColdStartSegmentation

        return YOLOOpenSetColdStartSegmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Prediction heads with lazy exports for standalone module execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detection_head import DetectionHead
    from .open_set_head import OpenSetHead
    from .segmentation_head import SegmentationHead

__all__ = ["DetectionHead", "OpenSetHead", "SegmentationHead"]


def __getattr__(name: str):
    if name == "DetectionHead":
        from .detection_head import DetectionHead

        return DetectionHead
    if name == "OpenSetHead":
        from .open_set_head import OpenSetHead

        return OpenSetHead
    if name == "SegmentationHead":
        from .segmentation_head import SegmentationHead

        return SegmentationHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

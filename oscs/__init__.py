"""YOLO Open-set Cold-start Segmentation package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .losses.composite_loss import CompositeLoss
    from .models.model import YOLOOpenSetColdStartSegmentation

__all__ = ["CompositeLoss", "YOLOOpenSetColdStartSegmentation"]


def __getattr__(name: str):
    if name == "CompositeLoss":
        from .losses.composite_loss import CompositeLoss

        return CompositeLoss
    if name == "YOLOOpenSetColdStartSegmentation":
        from .models.model import YOLOOpenSetColdStartSegmentation

        return YOLOOpenSetColdStartSegmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Loss modules with a lazy public export."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .composite_loss import CompositeLoss

__all__ = ["CompositeLoss"]


def __getattr__(name: str):
    if name == "CompositeLoss":
        from .composite_loss import CompositeLoss

        return CompositeLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

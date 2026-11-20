"""Inference wrappers with a lazy public export."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictor import Predictor

__all__ = ["Predictor"]


def __getattr__(name: str):
    if name == "Predictor":
        from .predictor import Predictor

        return Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Training wrappers with a lazy public export."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task import TrainingTask

__all__ = ["TrainingTask"]


def __getattr__(name: str):
    if name == "TrainingTask":
        from .task import TrainingTask

        return TrainingTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

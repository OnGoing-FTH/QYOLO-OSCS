"""Thin training wrapper; data loading remains outside this package."""

from __future__ import annotations

from torch import Tensor, nn

from oscs.losses import CompositeLoss
from oscs.utils.types import BatchTargets, LossOutput, ModelOutput


class TrainingTask(nn.Module):
    def __init__(self, model: nn.Module, criterion: CompositeLoss) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, images: Tensor, targets: BatchTargets) -> tuple[ModelOutput, LossOutput]:
        outputs = self.model(images)
        return outputs, self.criterion(outputs, targets)

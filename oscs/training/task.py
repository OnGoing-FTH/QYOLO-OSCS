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


if __name__ == "__main__":
    import torch

    from oscs.models import YOLOOpenSetColdStartSegmentation

    task = TrainingTask(YOLOOpenSetColdStartSegmentation(num_classes=3, width=16), CompositeLoss())
    targets = BatchTargets(
        labels=torch.tensor([[0, 1, -1]]),
        boxes=torch.rand(1, 3, 4),
        masks=torch.randint(0, 2, (1, 3, 128, 128)),
    )
    _, loss_output = task(torch.randn(1, 3, 128, 128), targets)
    print({"loss": float(loss_output.total.detach()), "components": tuple(loss_output.components)})

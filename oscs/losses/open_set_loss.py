"""Open-set knownness loss with a replaceable cold-start extension point."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn

from oscs.utils.types import BatchTargets, ModelOutput


class OpenSetLoss(nn.Module):
    def __init__(self, weight: float = 0.5) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, outputs: ModelOutput, targets: BatchTargets) -> dict[str, Tensor]:
        knownness_target = outputs.knownness_logits.new_zeros(outputs.knownness_logits.shape)
        query_count = min(outputs.knownness_logits.shape[1], targets.labels.shape[1])
        knownness_target[:, :query_count] = targets.labels[:, :query_count].ge(0).to(knownness_target.dtype)
        knownness = functional.binary_cross_entropy_with_logits(outputs.knownness_logits, knownness_target)
        return {"open_set": knownness * self.weight}


if __name__ == "__main__":
    import torch

    from oscs.models import YOLOOpenSetColdStartSegmentation

    model = YOLOOpenSetColdStartSegmentation(num_classes=3, width=16)
    outputs = model(torch.randn(1, 3, 128, 128))
    targets = BatchTargets(
        labels=torch.tensor([[0, 1, -1]]),
        boxes=torch.rand(1, 3, 4),
        masks=torch.randint(0, 2, (1, 3, 128, 128)),
    )
    print({name: float(value.detach()) for name, value in OpenSetLoss()(outputs, targets).items()})

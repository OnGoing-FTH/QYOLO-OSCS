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

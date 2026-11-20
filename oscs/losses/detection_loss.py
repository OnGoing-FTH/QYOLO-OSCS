"""Detection losses for the initial dense one-to-one assignment baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from oscs.utils.types import BatchTargets, ModelOutput


class DetectionLoss(nn.Module):
    def __init__(self, classification_weight: float = 1.0, box_weight: float = 5.0) -> None:
        super().__init__()
        self.classification_weight = classification_weight
        self.box_weight = box_weight

    def forward(self, outputs: ModelOutput, targets: BatchTargets) -> dict[str, Tensor]:
        labels, boxes, valid = _matched_targets(outputs, targets)
        if not valid.any():
            zero = outputs.class_logits.sum() * 0.0
            return {"classification": zero, "box": zero}

        predicted_logits = outputs.class_logits[:, : labels.shape[1]]
        predicted_boxes = outputs.boxes[:, : boxes.shape[1]]
        classification = functional.cross_entropy(predicted_logits[valid], labels[valid])
        box = functional.smooth_l1_loss(predicted_boxes[valid], boxes[valid])
        return {
            "classification": classification * self.classification_weight,
            "box": box * self.box_weight,
        }


def _matched_targets(outputs: ModelOutput, targets: BatchTargets) -> tuple[Tensor, Tensor, Tensor]:
    """Match target slot i to prediction slot i for scaffold validation only.

    This is intentionally isolated so it can be replaced by a task-aligned
    assigner without changing model heads or public loss interfaces.
    """
    query_count = min(outputs.class_logits.shape[1], targets.labels.shape[1])
    labels = targets.labels[:, :query_count]
    boxes = targets.boxes[:, :query_count]
    valid = labels.ge(0)
    return labels, boxes, valid


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
    print({name: float(value.detach()) for name, value in DetectionLoss()(outputs, targets).items()})

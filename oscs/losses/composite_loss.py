"""Task-level loss composition for YOLO-OSCS."""

from __future__ import annotations

from torch import nn

from oscs.utils.types import BatchTargets, LossOutput, ModelOutput

from .detection_loss import DetectionLoss
from .open_set_loss import OpenSetLoss
from .segmentation_loss import SegmentationLoss


class CompositeLoss(nn.Module):
    def __init__(
        self,
        classification_weight: float = 1.0,
        box_weight: float = 5.0,
        mask_weight: float = 1.0,
        dice_weight: float = 1.0,
        open_set_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.detection = DetectionLoss(classification_weight, box_weight)
        self.segmentation = SegmentationLoss(mask_weight, dice_weight)
        self.open_set = OpenSetLoss(open_set_weight)

    def forward(self, outputs: ModelOutput, targets: BatchTargets) -> LossOutput:
        components = {}
        components.update(self.detection(outputs, targets))
        components.update(self.segmentation(outputs, targets))
        components.update(self.open_set(outputs, targets))
        total = sum(components.values())
        return LossOutput(total=total, components=components)

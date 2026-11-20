"""Mask BCE and Dice losses for matched segmentation instances."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from oscs.utils.types import BatchTargets, ModelOutput


class SegmentationLoss(nn.Module):
    def __init__(self, mask_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight

    def forward(self, outputs: ModelOutput, targets: BatchTargets) -> dict[str, Tensor]:
        query_count = min(outputs.instance_masks.shape[1], targets.masks.shape[1])
        valid = targets.labels[:, :query_count].ge(0)
        if not valid.any():
            zero = outputs.instance_masks.sum() * 0.0
            return {"mask": zero, "dice": zero}

        predicted_masks = outputs.instance_masks[:, :query_count]
        target_masks = targets.masks[:, :query_count].float()
        target_masks = functional.interpolate(
            target_masks.flatten(0, 1).unsqueeze(1),
            size=predicted_masks.shape[-2:],
            mode="nearest",
        ).squeeze(1).unflatten(0, target_masks.shape[:2])
        predicted_masks = predicted_masks[valid]
        target_masks = target_masks[valid]

        mask = functional.binary_cross_entropy_with_logits(predicted_masks, target_masks)
        probabilities = predicted_masks.sigmoid()
        intersection = (probabilities * target_masks).sum(dim=(-2, -1))
        denominator = probabilities.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1))
        dice = (1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))).mean()
        return {"mask": mask * self.mask_weight, "dice": dice * self.dice_weight}


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
    print({name: float(value.detach()) for name, value in SegmentationLoss()(outputs, targets).items()})

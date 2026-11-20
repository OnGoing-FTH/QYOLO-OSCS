"""Dense detection predictions shared by training and inference."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.box_regressor = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        class_logits = self.classifier(features).flatten(2).transpose(1, 2)
        boxes = self.box_regressor(features).sigmoid().flatten(2).transpose(1, 2)
        return class_logits, boxes


if __name__ == "__main__":
    head = DetectionHead(in_channels=32, num_classes=3)
    class_logits, boxes = head(torch.randn(1, 32, 8, 8))
    print({"class_logits": tuple(class_logits.shape), "boxes": tuple(boxes.shape)})

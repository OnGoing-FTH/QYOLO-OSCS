"""Inference decoder for the shared model output contract."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from oscs.utils.types import PredictionBatch


class Predictor:
    def __init__(self, model: nn.Module, confidence_threshold: float = 0.25, top_k: int = 100) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, images: Tensor) -> PredictionBatch:
        outputs = self.model(images)
        class_scores, labels = outputs.class_logits.softmax(dim=-1).max(dim=-1)
        knownness = outputs.knownness_logits.sigmoid()
        scores = class_scores * knownness
        top_k = min(self.top_k, scores.shape[1])
        scores, indices = scores.topk(top_k, dim=1)
        labels = labels.gather(1, indices)
        boxes = outputs.boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, 4))
        masks = outputs.instance_masks.gather(
            1,
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *outputs.instance_masks.shape[-2:]),
        ).sigmoid()
        knownness = knownness.gather(1, indices)
        keep = scores.ge(self.confidence_threshold)
        return PredictionBatch(
            scores=scores * keep,
            labels=labels.masked_fill(~keep, -1),
            boxes=boxes,
            masks=masks,
            knownness=knownness,
        )


if __name__ == "__main__":
    from oscs.models import YOLOOpenSetColdStartSegmentation

    predictor = Predictor(YOLOOpenSetColdStartSegmentation(num_classes=3, width=16))
    predictions = predictor(torch.randn(1, 3, 128, 128))
    print({name: tuple(getattr(predictions, name).shape) for name in predictions.__dataclass_fields__})

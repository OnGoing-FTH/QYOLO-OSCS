"""Unified YOLO Open-set Cold-start Segmentation model."""

from __future__ import annotations

from torch import Tensor, nn

from oscs.utils.types import ModelOutput

from .backbone import YOLOv11Backbone
from .heads import DetectionHead, OpenSetHead, SegmentationHead
from .neck import PANNeck


class YOLOOpenSetColdStartSegmentation(nn.Module):
    """One model contract for both training and inference.

    The model emits raw dense predictions. Loss matching is deliberately kept
    outside the network so target assignment can evolve independently.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        width: int = 32,
        depth: int = 1,
        mask_prototypes: int = 32,
        embedding_dim: int = 128,
        use_pixel_unshuffle_stem: bool = True,
        pixel_unshuffle_factor: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = YOLOv11Backbone(
            in_channels=in_channels,
            width=width,
            depth=depth,
            use_pixel_unshuffle_stem=use_pixel_unshuffle_stem,
            pixel_unshuffle_factor=pixel_unshuffle_factor,
        )
        self.neck = PANNeck(self.backbone.channels, depth=depth)
        feature_channels = self.neck.channels[0]
        self.detection_head = DetectionHead(feature_channels, num_classes)
        self.segmentation_head = SegmentationHead(feature_channels, mask_prototypes)
        self.open_set_head = OpenSetHead(feature_channels, embedding_dim)

    def forward(self, images: Tensor) -> ModelOutput:
        features = self.neck(self.backbone(images))
        prediction_feature = features[0]
        class_logits, boxes = self.detection_head(prediction_feature)
        mask_coefficients, mask_prototypes, instance_masks = self.segmentation_head(prediction_feature)
        knownness_logits, embeddings = self.open_set_head(prediction_feature)
        return ModelOutput(
            class_logits=class_logits,
            boxes=boxes,
            mask_coefficients=mask_coefficients,
            mask_prototypes=mask_prototypes,
            instance_masks=instance_masks,
            knownness_logits=knownness_logits,
            embeddings=embeddings,
        )


if __name__ == "__main__":
    import torch

    model = YOLOOpenSetColdStartSegmentation(num_classes=3, width=16)
    outputs = model(torch.randn(1, 3, 128, 128))
    print({name: tuple(getattr(outputs, name).shape) for name in outputs.__dataclass_fields__})

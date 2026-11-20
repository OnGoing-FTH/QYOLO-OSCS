"""Shared typed contracts between model, loss, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass
class BatchTargets:
    """Dense placeholder targets for the scaffold's one-to-one loss baseline.

    Shapes:
        labels: [batch, instances], where -1 marks padded instances.
        boxes: [batch, instances, 4], normalized xyxy boxes.
        masks: [batch, instances, height, width], binary masks.
    """

    labels: Tensor
    boxes: Tensor
    masks: Tensor


@dataclass
class ModelOutput:
    """Raw shared forward output used by both training and inference."""

    class_logits: Tensor
    boxes: Tensor
    mask_coefficients: Tensor
    mask_prototypes: Tensor
    instance_masks: Tensor
    knownness_logits: Tensor
    embeddings: Tensor


@dataclass
class LossOutput:
    total: Tensor
    components: Mapping[str, Tensor]


@dataclass
class MultiResolutionOutput:
    """Aligned scale branches and their adaptive fused representation."""

    aligned_features: Mapping[int, Tensor]
    fusion_weights: Tensor
    fused: Tensor


@dataclass
class PredictionBatch:
    scores: Tensor
    labels: Tensor
    boxes: Tensor
    masks: Tensor
    knownness: Tensor

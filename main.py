"""Composition root for YOLO Open-set Cold-start Segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from oscs import CompositeLoss, YOLOOpenSetColdStartSegmentation
from oscs.inference import Predictor
from oscs.training import TrainingTask
from oscs.utils import BatchTargets


def build_components(config_path: str | Path = "configs/oscs_yolov11_seg.yaml") -> tuple[TrainingTask, Predictor]:
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model = YOLOOpenSetColdStartSegmentation(**config["model"])
    criterion = CompositeLoss(**config["loss"])
    task = TrainingTask(model, criterion)
    predictor = Predictor(model, **config["inference"])
    return task, predictor


def main() -> None:
    task, predictor = build_components()
    images = torch.randn(2, 3, 128, 128)
    targets = BatchTargets(
        labels=torch.tensor([[0, 1, -1], [2, -1, -1]]),
        boxes=torch.rand(2, 3, 4),
        masks=torch.randint(0, 2, (2, 3, 128, 128)),
    )
    task.train()
    _, losses = task(images, targets)
    task.eval()
    predictions = predictor(images)
    print({"loss": float(losses.total.detach()), "prediction_shape": tuple(predictions.boxes.shape)})


if __name__ == "__main__":
    main()

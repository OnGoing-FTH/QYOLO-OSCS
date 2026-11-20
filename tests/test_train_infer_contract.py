import torch

from main import build_components
from oscs.utils import BatchTargets


def test_training_and_inference_share_the_same_model() -> None:
    task, predictor = build_components()
    assert task.model is predictor.model

    images = torch.randn(1, 3, 128, 128)
    targets = BatchTargets(
        labels=torch.tensor([[0, -1]]),
        boxes=torch.rand(1, 2, 4),
        masks=torch.randint(0, 2, (1, 2, 128, 128)),
    )
    _, losses = task(images, targets)
    predictions = predictor(images)

    assert torch.isfinite(losses.total)
    assert predictions.boxes.shape[0] == images.shape[0]

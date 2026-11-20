import torch

from oscs.losses import CompositeLoss
from oscs.models import YOLOOpenSetColdStartSegmentation
from oscs.utils import BatchTargets


def test_composite_loss_is_scalar_and_differentiable() -> None:
    model = YOLOOpenSetColdStartSegmentation(num_classes=3, width=16)
    outputs = model(torch.randn(2, 3, 128, 128))
    targets = BatchTargets(
        labels=torch.tensor([[0, 1, -1], [2, -1, -1]]),
        boxes=torch.rand(2, 3, 4),
        masks=torch.randint(0, 2, (2, 3, 128, 128)),
    )

    loss_output = CompositeLoss()(outputs, targets)
    loss_output.total.backward()

    assert loss_output.total.ndim == 0
    assert set(loss_output.components) == {"classification", "box", "mask", "dice", "open_set"}

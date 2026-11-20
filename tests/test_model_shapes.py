import torch

from oscs.models import YOLOOpenSetColdStartSegmentation


def test_model_produces_aligned_dense_predictions() -> None:
    model = YOLOOpenSetColdStartSegmentation(num_classes=3, width=16, embedding_dim=24)
    outputs = model(torch.randn(2, 3, 128, 128))

    assert outputs.class_logits.shape[:2] == outputs.boxes.shape[:2]
    assert outputs.class_logits.shape[:2] == outputs.mask_coefficients.shape[:2]
    assert outputs.class_logits.shape[:2] == outputs.knownness_logits.shape[:2]
    assert outputs.instance_masks.shape[:2] == outputs.class_logits.shape[:2]
    assert outputs.class_logits.shape[-1] == 3
    assert outputs.embeddings.shape[-1] == 24

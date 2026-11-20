"""Open-set confidence and instance embedding predictions."""

from __future__ import annotations

from torch import Tensor, nn


class OpenSetHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.knownness = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.embedding = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        knownness_logits = self.knownness(features).flatten(2).squeeze(1)
        embeddings = self.embedding(features).flatten(2).transpose(1, 2)
        return knownness_logits, embeddings

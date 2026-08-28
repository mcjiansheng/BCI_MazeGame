"""EEG-Conformer baseline adapted from Song et al. (TNSRE 2023)."""

from __future__ import annotations

import torch
from torch import nn

from .layers import TransformerBlock


class EEGConformer(nn.Module):
    """Spatial-temporal convolution followed by a Transformer encoder.

    Input shape: ``batch x 1 x channels x time``.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1000,
        n_classes: int = 4,
        embed_dim: int = 40,
        depth: int = 6,
        num_heads: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=(1, 25), bias=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(n_channels, 1), bias=True),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.encoder = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, dropout) for _ in range(depth)]
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            feature = self.patch_embedding(dummy)
            flattened_features = int(feature.numel())
        self.classifier = nn.Sequential(
            nn.Linear(flattened_features, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(self.patch_embedding(x))
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder(x)
        return self.classifier(x.flatten(1))


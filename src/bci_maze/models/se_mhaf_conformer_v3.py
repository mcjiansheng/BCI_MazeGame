"""Paper-aligned SE-MHAF-Conformer for motor-imagery EEG decoding.

This implementation follows Chapter 3 of the project thesis: three temporal
scales, depthwise spatial filtering, per-scale squeeze/excitation, six MHAF
encoder blocks, and a flattened two-layer classifier.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SamePadConv2d(nn.Conv2d):
    """Conv2d with explicit asymmetric same padding for even temporal kernels."""

    def __init__(self, *args, temporal_kernel: int, **kwargs):
        super().__init__(*args, kernel_size=(1, temporal_kernel), padding=0, **kwargs)
        total = temporal_kernel - 1
        self.left = total // 2
        self.right = total - self.left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self.left, self.right, 0, 0)))


class MultiScaleTemporalFeatures(nn.Module):
    def __init__(self, filters: int = 40, kernels: tuple[int, ...] = (64, 32, 16)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    SamePadConv2d(1, filters, temporal_kernel=kernel, bias=True),
                    nn.BatchNorm2d(filters),
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class SpatialFeatures(nn.Module):
    def __init__(self, in_features: int, n_channels: int, multiplier: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                in_features,
                in_features * multiplier,
                kernel_size=(n_channels, 1),
                groups=in_features,
                bias=True,
            ),
            nn.BatchNorm2d(in_features * multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.network = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.network(x.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class PerScaleEnhancement(nn.Module):
    def __init__(self, channels_per_scale: int, scales: int, reduction: int, dropout: float):
        super().__init__()
        self.scales = scales
        self.se_blocks = nn.ModuleList(
            [SqueezeExcitation(channels_per_scale, reduction) for _ in range(scales)]
        )
        self.pool = nn.AvgPool2d((1, 4), stride=(1, 4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.scales, dim=1)
        enhanced = torch.cat(
            [se(chunk) for se, chunk in zip(self.se_blocks, chunks)], dim=1
        )
        return self.dropout(self.pool(enhanced))


class PaperMHAF(nn.Module):
    """Multi-head attention with cross-layer score reuse and head fusion."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, alpha: float = 0.1):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.alpha = alpha
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.head_weights = nn.Parameter(torch.empty(num_heads, num_heads))
        nn.init.xavier_uniform_(self.head_weights)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return projection(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, previous_scores: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._heads(x, self.query)
        key = self._heads(x, self.key)
        value = self._heads(x, self.value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if previous_scores is not None:
            scores = scores + self.alpha * previous_scores.detach()
        attention = self.dropout(scores.softmax(dim=-1))
        heads = attention @ value

        # Equation (3-10): normalize each column over its source-head axis.
        mixing = self.head_weights.softmax(dim=0)
        heads = torch.einsum("oi,bind->bond", mixing, heads)
        batch, _, tokens, _ = heads.shape
        return heads.transpose(1, 2).reshape(batch, tokens, -1), scores


class PaperMHAFBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, expansion: int = 4):
        super().__init__()
        self.attention = PaperMHAF(embed_dim, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, previous_scores: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(x, previous_scores)
        x = self.attention_norm(x + self.attention_dropout(attended))
        x = self.feed_forward_norm(x + self.feed_forward_dropout(self.feed_forward(x)))
        return x, scores


class SEMHAFConformerV3(nn.Module):
    """Faithful implementation of the thesis SE-MHAF-Conformer architecture.

    Input is ``batch x 1 x channels x 1000 samples``. Defaults reproduce the
    thesis configuration: F1=40, D=2, embed=240, depth=6, heads=10.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1000,
        n_classes: int = 4,
        temporal_filters: int = 40,
        spatial_multiplier: int = 2,
        kernels: tuple[int, ...] = (64, 32, 16),
        depth: int = 6,
        num_heads: int = 10,
        se_reduction: int = 4,
        feed_forward_expansion: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        scales = len(kernels)
        temporal_channels = scales * temporal_filters
        embed_dim = temporal_channels * spatial_multiplier
        self.temporal = MultiScaleTemporalFeatures(temporal_filters, kernels)
        self.spatial = SpatialFeatures(
            temporal_channels, n_channels, spatial_multiplier, dropout
        )
        self.spatio_temporal = nn.Sequential(
            SamePadConv2d(
                embed_dim,
                embed_dim,
                temporal_kernel=16,
                groups=embed_dim,
                bias=True,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
        )
        self.enhancement = PerScaleEnhancement(
            temporal_filters * spatial_multiplier,
            scales,
            se_reduction,
            dropout,
        )
        self.encoder = nn.ModuleList(
            [
                PaperMHAFBlock(
                    embed_dim, num_heads, dropout, feed_forward_expansion
                )
                for _ in range(depth)
            ]
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            features = self._extract_features(dummy)
            flattened_features = features.shape[1] * features.shape[2]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, n_times // 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(n_times // 4, n_classes),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.spatio_temporal(x)
        x = self.enhancement(x)
        return x.squeeze(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._extract_features(x)
        previous_scores = None
        for block in self.encoder:
            x, previous_scores = block(x, previous_scores)
        return self.classifier(x)

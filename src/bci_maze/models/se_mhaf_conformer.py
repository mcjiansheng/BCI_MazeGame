"""SE-MHAF-Conformer implementation for four-class motor imagery."""

from __future__ import annotations

import math

import torch
from torch import nn

from .layers import Conv2dWithConstraint


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.network = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.network(x.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class TemporalSpatialBranch(nn.Module):
    def __init__(self, n_channels: int, temporal_kernel: int, filters: int, multiplier: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                1,
                filters,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(filters),
            Conv2dWithConstraint(
                filters,
                filters * multiplier,
                kernel_size=(n_channels, 1),
                groups=filters,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(filters * multiplier),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MHAF(nn.Module):
    """Attention with layer-to-layer correlation and learnable head mixing."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.head_mix = nn.Parameter(torch.eye(num_heads))
        self.correlation_scale = nn.Parameter(torch.tensor(-2.0))
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, previous_attention: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, embed = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if previous_attention is not None:
            scores = scores + torch.sigmoid(self.correlation_scale) * previous_attention
        attention = scores.softmax(dim=-1)
        dropped_attention = self.attention_dropout(attention)
        heads = torch.matmul(dropped_attention, v)
        mixing = self.head_mix.softmax(dim=-1)
        heads = torch.einsum("ij,bjnd->bind", mixing, heads)
        output = heads.transpose(1, 2).reshape(batch, tokens, embed)
        return self.projection(output), attention


class MHAFEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MHAF(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, previous_attention: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(self.norm1(x), previous_attention)
        x = x + self.dropout1(attended)
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x, attention


class SEMHAFConformer(nn.Module):
    """Multi-scale temporal CNN + SE + MHAF Transformer.

    Input shape: ``batch x 1 x channels x time``.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1000,
        n_classes: int = 4,
        branch_filters: int = 16,
        spatial_multiplier: int = 2,
        embed_dim: int = 120,
        depth: int = 6,
        num_heads: int = 10,
        dropout: float = 0.4,
    ):
        super().__init__()
        kernels = (63, 31, 15)
        self.branches = nn.ModuleList(
            [
                TemporalSpatialBranch(n_channels, kernel, branch_filters, spatial_multiplier)
                for kernel in kernels
            ]
        )
        branch_output = len(kernels) * branch_filters * spatial_multiplier
        self.fusion = nn.Sequential(
            nn.Conv2d(
                branch_output,
                branch_output,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=branch_output,
                bias=False,
            ),
            nn.Conv2d(branch_output, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            SqueezeExcitation(embed_dim),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        self.encoder = nn.ModuleList(
            [MHAFEncoderBlock(embed_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fusion(x).squeeze(2).transpose(1, 2)
        previous_attention = None
        for block in self.encoder:
            x, previous_attention = block(x, previous_attention)
        return self.classifier(x.mean(dim=1))


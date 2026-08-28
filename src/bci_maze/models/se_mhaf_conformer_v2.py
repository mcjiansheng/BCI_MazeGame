"""Optimized SE-MHAF-Conformer with an explicit FBC-logvar branch."""

from __future__ import annotations

import math

import torch
from torch import nn

from .fbcnet import FBCNet
from .se_mhaf_conformer import SqueezeExcitation, TemporalSpatialBranch


class StableMHAF(nn.Module):
    """MHAF with identity-preserving head fusion and zero-gated correlation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.head_mix_logits = nn.Parameter(torch.zeros(num_heads, num_heads))
        self.head_mix_scale = nn.Parameter(torch.tensor(0.0))
        self.correlation_scale = nn.Parameter(torch.tensor(0.0))
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
            # Convert probabilities back to a centered, normalized log-prior.
            prior = torch.log(previous_attention.detach().clamp_min(1e-6))
            prior = prior - prior.mean(dim=-1, keepdim=True)
            prior = prior / prior.std(dim=-1, keepdim=True).clamp_min(1e-5)
            scores = scores + torch.tanh(self.correlation_scale) * prior
        attention = scores.softmax(dim=-1)
        heads = torch.matmul(self.attention_dropout(attention), v)

        # The residual path is exactly identity at initialization. Learned
        # cross-head mixing is introduced only when its zero-initialized gate
        # moves away from zero.
        mixing = self.head_mix_logits.softmax(dim=-1)
        mixed_heads = torch.einsum("ij,bjnd->bind", mixing, heads)
        heads = heads + torch.tanh(self.head_mix_scale) * mixed_heads
        output = heads.transpose(1, 2).reshape(batch, tokens, embed)
        return self.projection(output), attention


class StableMHAFBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, expansion: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = StableMHAF(embed_dim, num_heads, dropout)
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


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.score(x).softmax(dim=1)
        return torch.sum(weights * x, dim=1)


class SEMHAFConformerV2(nn.Module):
    """Explicit FBCNet branch plus a compact, stabilized SE-MHAF branch.

    Input shape is ``batch x 9 bands x channels x 1000 samples``. The FBC
    branch preserves the strong filter-bank/log-variance prior. The compact
    Conformer branch operates on the reconstructed broadband view, and both
    logits are combined as a zero-initialized residual correction of the FBC
    logits, so the optimized path starts from an exact FBCNet predictor.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1000,
        n_classes: int = 4,
        n_bands: int = 9,
        branch_filters: int = 8,
        spatial_multiplier: int = 2,
        embed_dim: int = 48,
        depth: int = 3,
        num_heads: int = 6,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.fbc_branch = FBCNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            n_bands=n_bands,
            spatial_filters=32,
            temporal_segments=4,
        )
        kernels = (63, 31, 15)
        self.branches = nn.ModuleList(
            [
                TemporalSpatialBranch(
                    n_channels, kernel, branch_filters, spatial_multiplier
                )
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
            SqueezeExcitation(embed_dim, reduction=6),
            nn.Dropout(dropout),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        n_tokens = (n_times // 4) // 4
        self.position = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.encoder = nn.ModuleList(
            [StableMHAFBlock(embed_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.pool = AttentionPool(embed_dim)
        self.mhaf_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes),
        )
        # The fused predictor is exactly FBCNet at initialization. Auxiliary
        # branch losses train MHAF even while this correction gate is zero.
        self.residual_scale = nn.Parameter(torch.zeros(n_classes))

    def forward_branches(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fbc_logits = self.fbc_branch(x)
        broadband = x.mean(dim=1, keepdim=True)
        features = torch.cat([branch(broadband) for branch in self.branches], dim=1)
        tokens = self.fusion(features).squeeze(2).transpose(1, 2)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        previous_attention = None
        for block in self.encoder:
            tokens, previous_attention = block(tokens, previous_attention)
        mhaf_logits = self.mhaf_classifier(self.pool(tokens))
        scale = torch.tanh(self.residual_scale).unsqueeze(0)
        return fbc_logits, mhaf_logits, scale

    @staticmethod
    def fuse_logits(
        fbc_logits: torch.Tensor, mhaf_logits: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return fbc_logits + scale * mhaf_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fbc_logits, mhaf_logits, scale = self.forward_branches(x)
        return self.fuse_logits(fbc_logits, mhaf_logits, scale)

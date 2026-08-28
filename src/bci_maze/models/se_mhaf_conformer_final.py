"""Data-efficient SE-MHAF-Conformer with a protected FBCNet backbone."""

from __future__ import annotations

import torch
from torch import nn

from .fbcnet import FBCNet
from .se_mhaf_conformer import SqueezeExcitation, TemporalSpatialBranch
from .se_mhaf_conformer_v2 import AttentionPool, StableMHAFBlock


class BandSqueezeExcitation(nn.Module):
    """Apply independent SE recalibration to each filter-bank view."""

    def __init__(self, n_bands: int, features: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, features // reduction)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(features, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, features),
                    nn.Sigmoid(),
                )
                for _ in range(n_bands)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch x bands x temporal_segments x spatial_filters
        outputs = []
        for band, block in enumerate(self.blocks):
            view = x[:, band]
            weights = block(view.mean(dim=1)).unsqueeze(1)
            outputs.append(view * weights)
        return torch.stack(outputs, dim=1)


class SEMHAFConformerFinal(nn.Module):
    """Protected FBC logits plus an SE-MHAF residual correction.

    The backbone is trained independently and can then be frozen. With the
    zero-initialized residual gate, the complete model is exactly FBCNet before
    residual training, providing a measurable non-regression invariant.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1000,
        n_classes: int = 4,
        n_bands: int = 9,
        spatial_filters: int = 32,
        temporal_segments: int = 4,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 8,
        dropout: float = 0.35,
        use_temporal_expert: bool = True,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.spatial_filters = spatial_filters
        self.temporal_segments = temporal_segments
        self.fbc_branch = FBCNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            n_bands=n_bands,
            spatial_filters=spatial_filters,
            temporal_segments=temporal_segments,
        )
        self.band_se = BandSqueezeExcitation(n_bands, spatial_filters)
        self.embedding = nn.Sequential(
            nn.Linear(spatial_filters, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.band_position = nn.Parameter(torch.zeros(1, n_bands, 1, embed_dim))
        self.segment_position = nn.Parameter(
            torch.zeros(1, 1, temporal_segments, embed_dim)
        )
        nn.init.trunc_normal_(self.band_position, std=0.02)
        nn.init.trunc_normal_(self.segment_position, std=0.02)
        self.encoder = nn.ModuleList(
            [StableMHAFBlock(embed_dim, num_heads, dropout, expansion=2) for _ in range(depth)]
        )
        token_count = n_bands * temporal_segments
        self.residual_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Flatten(),
            nn.Linear(token_count * embed_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self.use_temporal_expert = use_temporal_expert
        if not use_temporal_expert:
            self.residual_scale = nn.Parameter(torch.zeros(n_classes))
            return

        kernels = (63, 31, 15)
        branch_filters = 8
        spatial_multiplier = 2
        temporal_embed = 48
        self.temporal_branches = nn.ModuleList(
            [
                TemporalSpatialBranch(
                    n_channels, kernel, branch_filters, spatial_multiplier
                )
                for kernel in kernels
            ]
        )
        temporal_features = len(kernels) * branch_filters * spatial_multiplier
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(
                temporal_features,
                temporal_features,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=temporal_features,
                bias=False,
            ),
            nn.Conv2d(temporal_features, temporal_embed, kernel_size=1, bias=False),
            nn.BatchNorm2d(temporal_embed),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            SqueezeExcitation(temporal_embed, reduction=6),
            nn.Dropout(0.4),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
        )
        temporal_tokens = (n_times // 4) // 4
        self.temporal_position = nn.Parameter(
            torch.zeros(1, temporal_tokens, temporal_embed)
        )
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        self.temporal_encoder = nn.ModuleList(
            [StableMHAFBlock(temporal_embed, 6, 0.4, expansion=2) for _ in range(3)]
        )
        self.temporal_pool = AttentionPool(temporal_embed)
        self.temporal_classifier = nn.Sequential(
            nn.LayerNorm(temporal_embed),
            nn.Linear(temporal_embed, n_classes),
        )
        self.expert_mix_logits = nn.Parameter(torch.zeros(2))
        self.residual_scale = nn.Parameter(torch.zeros(n_classes))

    def logvar_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fbc_branch.spatial(x)
        batch, _, _, samples = x.shape
        x = x.reshape(
            batch,
            self.n_bands,
            self.spatial_filters,
            self.temporal_segments,
            samples // self.temporal_segments,
        )
        x = torch.log(torch.clamp(x.var(dim=-1, unbiased=True), 1e-6, 1e6))
        return x.permute(0, 1, 3, 2).contiguous()

    def forward_branches(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.logvar_features(x)
        fbc_logits = self.fbc_branch.classifier(features.permute(0, 1, 3, 2).flatten(1))
        tokens = self.embedding(self.band_se(features))
        tokens = tokens + self.band_position + self.segment_position
        tokens = tokens.flatten(1, 2)
        previous_attention = None
        for block in self.encoder:
            tokens, previous_attention = block(tokens, previous_attention)
        logvar_logits = self.residual_classifier(tokens)

        if not self.use_temporal_expert:
            scale = torch.tanh(self.residual_scale).unsqueeze(0)
            return fbc_logits, logvar_logits, scale

        broadband = x.mean(dim=1, keepdim=True)
        temporal = torch.cat(
            [branch(broadband) for branch in self.temporal_branches], dim=1
        )
        temporal = self.temporal_fusion(temporal).squeeze(2).transpose(1, 2)
        temporal = temporal + self.temporal_position[:, : temporal.shape[1]]
        previous_attention = None
        for block in self.temporal_encoder:
            temporal, previous_attention = block(temporal, previous_attention)
        temporal_logits = self.temporal_classifier(self.temporal_pool(temporal))

        expert_mix = self.expert_mix_logits.softmax(dim=0)
        residual_logits = expert_mix[0] * logvar_logits + expert_mix[1] * temporal_logits
        scale = torch.tanh(self.residual_scale).unsqueeze(0)
        return fbc_logits, residual_logits, scale

    @staticmethod
    def fuse_logits(
        fbc_logits: torch.Tensor,
        residual_logits: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return fbc_logits + scale * residual_logits

    def freeze_backbone(self) -> None:
        for parameter in self.fbc_branch.parameters():
            parameter.requires_grad_(False)
        self.fbc_branch.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.fbc_branch.parameters()):
            self.fbc_branch.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fbc, residual, scale = self.forward_branches(x)
        return self.fuse_logits(fbc, residual, scale)

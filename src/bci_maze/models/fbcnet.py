"""FBCNet baseline following Mane et al. (EMBC 2021)."""

from __future__ import annotations

import torch
from torch import nn

from .layers import Conv2dWithConstraint, LinearWithConstraint


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FBCNet(nn.Module):
    """Filter-bank spatial convolution and segmented log-variance.

    Input shape: ``batch x bands x channels x time``.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1000,
        n_classes: int = 4,
        n_bands: int = 9,
        spatial_filters: int = 32,
        temporal_segments: int = 4,
    ):
        super().__init__()
        if n_times % temporal_segments:
            raise ValueError("n_times must be divisible by temporal_segments")
        self.n_bands = n_bands
        self.spatial_filters = spatial_filters
        self.temporal_segments = temporal_segments
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(
                n_bands,
                spatial_filters * n_bands,
                kernel_size=(n_channels, 1),
                groups=n_bands,
                bias=True,
                max_norm=2.0,
            ),
            nn.BatchNorm2d(spatial_filters * n_bands),
            Swish(),
        )
        self.classifier = LinearWithConstraint(
            spatial_filters * n_bands * temporal_segments,
            n_classes,
            max_norm=0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        batch, features, _, samples = x.shape
        x = x.reshape(batch, features, self.temporal_segments, samples // self.temporal_segments)
        # torch.var's unbiased estimator matches the authors' LogVarLayer.
        x = torch.log(torch.clamp(x.var(dim=-1, unbiased=True), 1e-6, 1e6))
        return self.classifier(x.flatten(1))

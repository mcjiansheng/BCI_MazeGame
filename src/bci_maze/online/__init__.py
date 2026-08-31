"""Streaming inference for real-time BCI maze control.

This package contains the online-side counterparts of the offline training
pipeline: causal filtering, frozen normalization, sliding-window decoding and
a hysteresis decision policy that turns per-window class probabilities into
discrete maze commands.
"""

from .decoder import (
    CausalBandpass,
    HysteresisPolicy,
    LabelMap,
    OnlineDecoder,
    load_label_map,
)

__all__ = [
    "CausalBandpass",
    "HysteresisPolicy",
    "LabelMap",
    "OnlineDecoder",
    "load_label_map",
]

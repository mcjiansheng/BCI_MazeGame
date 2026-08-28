"""Model registry for the three BCIC2a experiments."""

from .eeg_conformer import EEGConformer
from .fbcnet import FBCNet
from .se_mhaf_conformer import SEMHAFConformer
from .se_mhaf_conformer_v2 import SEMHAFConformerV2
from .se_mhaf_conformer_v3 import SEMHAFConformerV3
from .se_mhaf_conformer_final import SEMHAFConformerFinal


def build_model(name: str, n_channels: int = 22, n_times: int = 1000, n_classes: int = 4):
    normalized = name.lower().replace("-", "_")
    if normalized == "eeg_conformer":
        return EEGConformer(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    if normalized == "fbcnet":
        return FBCNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    if normalized in {"se_mhaf_conformer", "semhaf_conformer"}:
        return SEMHAFConformer(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    if normalized in {"se_mhaf_conformer_v2", "semhaf_conformer_v2"}:
        return SEMHAFConformerV2(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    if normalized in {"se_mhaf_conformer_final", "semhaf_conformer_final"}:
        return SEMHAFConformerFinal(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    if normalized in {"se_mhaf_conformer_final_logvar", "semhaf_conformer_final_logvar"}:
        return SEMHAFConformerFinal(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            use_temporal_expert=False,
        )
    if normalized in {"se_mhaf_conformer_v3_compact", "semhaf_conformer_v3_compact"}:
        return SEMHAFConformerV3(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            temporal_filters=8,
            depth=3,
            num_heads=6,
            dropout=0.4,
        )
    if normalized in {
        "se_mhaf_conformer_v3",
        "semhaf_conformer_v3",
        "se_mhaf_conformer_paper",
        "se_mhaf_conformer_v3_raw",
    }:
        return SEMHAFConformerV3(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    raise ValueError(f"Unknown model: {name}")


__all__ = [
    "EEGConformer",
    "FBCNet",
    "SEMHAFConformer",
    "SEMHAFConformerV2",
    "SEMHAFConformerV3",
    "SEMHAFConformerFinal",
    "build_model",
]

"""Portable 16-channel SE-MHAF model and identical offline/online preprocessing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np
import torch

from ..models import build_model
from ..preprocessing import FILTER_BANK_BANDS, make_filter_bank
from ..stopping import defer_stop
from .source import DEFAULT_CHANNEL_NAMES

MODEL_NAME = "se_mhaf_conformer_final_logvar"
CLASSES = ("left_hand", "right_hand")
FORMAT = "lk_semhaf16_v1"
# Dataset IV-2a Fig. 3 and MOABB's BNCI2014-001 loader agree on this order.
MONTAGE_SOURCE = "https://www.bbci.de/competition/iv/desc_2a.pdf"
MONTAGE_ORDER_SOURCE = "https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/bnci_2014.py"
BCIC2A_CHANNEL_NAMES = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
BCIC2A_GDF_NAMES = (
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5", "EEG-C3",
    "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9", "EEG-10", "EEG-11",
    "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
)


@dataclass(frozen=True)
class SEMHAFConfig:
    sample_rate: int = 250
    epoch_seconds: float = 4.0


def resolve_device(device: str = "auto") -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def validate_epochs(epochs: np.ndarray, sample_rate: int, channel_names: Sequence[str]) -> np.ndarray:
    values = np.asarray(epochs, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if sample_rate != 250:
        raise ValueError(f"SE-MHAF requires 250 Hz, got {sample_rate}")
    if tuple(channel_names) != DEFAULT_CHANNEL_NAMES:
        raise ValueError(f"SE-MHAF channel order must be {DEFAULT_CHANNEL_NAMES}")
    if values.ndim != 3 or values.shape[1:] != (16, 1000) or not len(values):
        raise ValueError(f"SE-MHAF requires non-empty (*, 16, 1000) epochs, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("EEG contains NaN or infinity")
    return values


def public_channel_indices(channel_names: Sequence[str]) -> list[int]:
    names = tuple(str(name) for name in channel_names)
    if names == BCIC2A_GDF_NAMES:
        names = BCIC2A_CHANNEL_NAMES
    else:
        names = tuple(name.removeprefix("EEG-") for name in names)
    if len(names) != 22 or len(set(names)) != 22 or set(names) != set(BCIC2A_CHANNEL_NAMES):
        raise ValueError("Unrecognized BCIC2a 22-channel montage; supply verified channel labels")
    return [names.index(name) for name in DEFAULT_CHANNEL_NAMES]


def filter_epochs(epochs: np.ndarray) -> np.ndarray:
    return _filter_validated_epochs(validate_epochs(epochs, 250, DEFAULT_CHANNEL_NAMES))


def _filter_validated_epochs(values: np.ndarray) -> np.ndarray:
    # CAR uses precisely the same 16 electrodes on both acquisition systems.
    referenced = values - values.mean(axis=1, keepdims=True)
    return make_filter_bank(referenced, sfreq=250.0, zero_phase=False)


def fit_normalization(filtered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(filtered, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (9, 16, 1000) or not len(values):
        raise ValueError("Normalization requires non-empty train x 9 x 16 x 1000 data")
    mean = values.mean(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    std = values.std(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def _preprocessing_contract() -> dict:
    return {
        "input_units": "uV", "reference": "CAR", "reference_channels": list(DEFAULT_CHANNEL_NAMES),
        "bands_hz": [list(band) for band in FILTER_BANK_BANDS],
        "filter": "Chebyshev-II", "passband_ripple_db": 3.0, "attenuation_db": 30.0,
        "transition_hz": 2.0, "causal": True, "state": "zero_at_each_window",
        "normalization": "training_trials_per_band_per_channel_mean_std",
    }


def preprocessing_metadata() -> dict:
    """Describe the algorithm and any additional notes stored with the model."""
    return _preprocessing_contract()


@dataclass
class SEMHAFBundle:
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    subject_id: str
    metadata: dict
    training_report: dict
    device: torch.device
    config: SEMHAFConfig = SEMHAFConfig()
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES
    classes: tuple[str, ...] = CLASSES

    @property
    def provenance(self) -> dict:
        return self.metadata.get("provenance", {})

    @property
    def training_files(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("training_files", []))

    def validate_input(self, epoch: np.ndarray, sample_rate: int, channel_names: Sequence[str]) -> np.ndarray:
        return validate_epochs(epoch, sample_rate, channel_names)

    def predict_proba(self, epoch: np.ndarray, sample_rate: int, channel_names: Sequence[str]) -> np.ndarray:
        values = self.validate_input(epoch, sample_rate, channel_names)
        filtered = _filter_validated_epochs(values)
        normalized = (filtered - self.mean) / self.std
        self.model.eval()
        with torch.inference_mode():
            return self.model(torch.from_numpy(normalized).to(self.device)).softmax(dim=1).cpu().numpy()


def new_bundle(*, device: str = "auto", subject_id: str = "public", metadata: dict | None = None) -> SEMHAFBundle:
    target = resolve_device(device)
    model = build_model(MODEL_NAME, n_channels=16, n_times=1000, n_classes=2).to(target)
    return SEMHAFBundle(model, np.zeros((1, 9, 16, 1), np.float32),
                        np.ones((1, 9, 16, 1), np.float32), subject_id,
                        metadata or {}, {}, target)


@defer_stop()
def save_semhaf_bundle(bundle: SEMHAFBundle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # JSON round-trip guarantees metadata uses primitives accepted by weights_only.
    payload = {
        "format": FORMAT, "model_name": MODEL_NAME,
        "config": asdict(bundle.config), "channel_names": list(bundle.channel_names),
        "classes": list(bundle.classes), "subject_id": bundle.subject_id,
        "preprocessing": json.loads(json.dumps(preprocessing_metadata())),
        "model": {key: value.detach().cpu() for key, value in bundle.model.state_dict().items()},
        "norm_mean": torch.from_numpy(bundle.mean.copy()), "norm_std": torch.from_numpy(bundle.std.copy()),
        "metadata": json.loads(json.dumps(bundle.metadata)),
        "training_report": json.loads(json.dumps(bundle.training_report)),
    }
    sidecar = destination.with_suffix(".json")
    with TemporaryDirectory(dir=destination.parent) as staging_dir:
        staging = Path(staging_dir)
        temporary_checkpoint = staging / "checkpoint.pt"
        temporary_sidecar = staging / "checkpoint.json"
        torch.save(payload, temporary_checkpoint)
        temporary_sidecar.write_text(json.dumps({
            key: value for key, value in payload.items() if key not in {"model", "norm_mean", "norm_std"}
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_checkpoint.replace(destination)
        temporary_sidecar.replace(sidecar)
    return destination


def load_semhaf_bundle(path: str | Path, device: str = "auto") -> SEMHAFBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT or checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError("Expected an lk_semhaf16_v1 checkpoint; pretrain/finetune with the LK scripts")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in asdict(SEMHAFConfig()).items()):
        raise ValueError("Checkpoint sample rate/window differs from the LK contract")
    if tuple(checkpoint.get("channel_names", [])) != DEFAULT_CHANNEL_NAMES or tuple(checkpoint.get("classes", [])) != CLASSES:
        raise ValueError("Checkpoint channel order or classes differ from the LK contract")
    preprocessing = checkpoint.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        raise ValueError("Checkpoint preprocessing must be a dictionary")
    for key, value in _preprocessing_contract().items():
        if preprocessing.get(key) != value:
            raise ValueError(f"Checkpoint preprocessing differs from the LK contract: {key}")
    bundle = new_bundle(device=device, subject_id=str(checkpoint["subject_id"]), metadata=checkpoint["metadata"])
    bundle.model.load_state_dict(checkpoint["model"], strict=True)
    bundle.mean = checkpoint["norm_mean"].numpy().astype(np.float32)
    bundle.std = checkpoint["norm_std"].numpy().astype(np.float32)
    if bundle.mean.shape != (1, 9, 16, 1) or bundle.std.shape != (1, 9, 16, 1):
        raise ValueError("Checkpoint normalization shape is invalid")
    if not np.isfinite(bundle.mean).all() or not np.isfinite(bundle.std).all() or (bundle.std <= 0).any():
        raise ValueError("Checkpoint normalization values are invalid")
    bundle.training_report = checkpoint["training_report"]
    bundle.model.eval()
    return bundle

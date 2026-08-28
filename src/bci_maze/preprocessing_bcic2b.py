"""Preprocessing for BCI Competition IV data set 2b (BNCI 004-2014)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from .preprocessing import _bandpass


EEG_CHANNEL_COUNT = 3
CHANNEL_NAMES = ("C3", "Cz", "C4")
CLASS_NAMES = ("left_hand", "right_hand")


@dataclass(frozen=True)
class BCIC2bPreprocessConfig:
    sfreq: float = 250.0
    epoch_start: float = 3.0
    epoch_end: float = 7.0
    low_freq: float = 4.0
    high_freq: float = 40.0
    filter_order: int = 6
    stop_attenuation_db: float = 60.0

    @property
    def n_times(self) -> int:
        return int(round((self.epoch_end - self.epoch_start) * self.sfreq))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bnci_mat(
    mat_path: str | Path,
    config: BCIC2bPreprocessConfig = BCIC2bPreprocessConfig(),
) -> dict[str, np.ndarray]:
    """Read a BNCI 004-2014 subject split and concatenate its sessions.

    ``BxxT.mat`` contains sessions 01T--03T and ``BxxE.mat`` contains
    sessions 04E--05E. MATLAB trial positions are one-based and point to the
    trial start; the motor-imagery interval used here is trial +3 s to +7 s.
    """
    mat_path = Path(mat_path)
    sessions = np.atleast_1d(
        loadmat(mat_path, squeeze_me=True, struct_as_record=False)["data"]
    )
    raw_trials, labels, artifacts, session_ids = [], [], [], []
    for session_index, session in enumerate(sessions):
        sfreq = float(session.fs)
        if not np.isclose(sfreq, config.sfreq):
            raise ValueError(f"Expected {config.sfreq} Hz, got {sfreq} Hz in {mat_path}")
        signal = np.asarray(session.X[:, :EEG_CHANNEL_COUNT], dtype=np.float32)
        session_labels = np.asarray(session.y, dtype=np.int64).reshape(-1) - 1
        session_artifacts = np.asarray(session.artifacts, dtype=bool).reshape(-1)
        trial_starts = np.asarray(session.trial, dtype=np.int64).reshape(-1) - 1
        if not (
            trial_starts.size == session_labels.size == session_artifacts.size
            and np.isin(session_labels, (0, 1)).all()
        ):
            raise ValueError(f"Invalid trials/labels/artifacts in {mat_path}, session {session_index}")

        offset = int(round(config.epoch_start * sfreq))
        trials = np.empty(
            (session_labels.size, EEG_CHANNEL_COUNT, config.n_times), dtype=np.float32
        )
        for trial_index, trial_start in enumerate(trial_starts):
            start = int(trial_start) + offset
            stop = start + config.n_times
            if stop > signal.shape[0]:
                raise ValueError(
                    f"Epoch {trial_index} exceeds session length in {mat_path}"
                )
            trials[trial_index] = np.nan_to_num(signal[start:stop].T, copy=False)
        raw_trials.append(trials)
        labels.append(session_labels)
        artifacts.append(session_artifacts)
        session_ids.append(np.full(session_labels.size, session_index, dtype=np.int8))

    raw_x = np.concatenate(raw_trials)
    filtered_x = _bandpass(
        raw_x,
        config.sfreq,
        config.low_freq,
        config.high_freq,
        config.filter_order,
        config.stop_attenuation_db,
    )
    return {
        "x": filtered_x,
        "raw_x": raw_x,
        "y": np.concatenate(labels),
        "artifact": np.concatenate(artifacts),
        "session_id": np.concatenate(session_ids),
        "channel_names": np.asarray(CHANNEL_NAMES),
        "sfreq": np.asarray(config.sfreq, dtype=np.float32),
    }


def preprocess_bcic2b(
    raw_dir: str | Path,
    output_dir: str | Path,
    subjects: Iterable[int] = range(1, 10),
    config: BCIC2bPreprocessConfig = BCIC2bPreprocessConfig(),
    overwrite: bool = False,
) -> list[Path]:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written, checksums = [], {}
    for subject in subjects:
        for split in ("T", "E"):
            stem = f"B{subject:02d}{split}"
            source = raw_dir / f"{stem}.mat"
            if not source.exists():
                raise FileNotFoundError(source)
            checksums[source.name] = _sha256(source)
            destination = output_dir / f"{stem}.npz"
            if overwrite or not destination.exists():
                np.savez_compressed(destination, **read_bnci_mat(source, config))
            written.append(destination)

    metadata = {
        "dataset": "BCI Competition IV 2b / BNCI 004-2014",
        "source": "https://bnci-horizon-2020.eu/database/data-sets",
        "license": "CC BY-ND 4.0",
        "file_prefix": "B",
        "classes": list(CLASS_NAMES),
        "channels": list(CHANNEL_NAMES),
        "config": asdict(config),
        "source_sha256": checksums,
        "files": [path.name for path in written],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return written


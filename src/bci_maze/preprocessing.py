"""BCI Competition IV 2a preprocessing.

The default epoch and broadband filter reproduce the public EEG-Conformer
preprocessing script: trial start + 2 s through + 6 s, 4--40 Hz, 250 Hz.
All 22 EEG channels are retained; the three EOG channels are never exposed to
the classifiers. Artifact flags are preserved so experiments can include or
exclude those trials explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import cheb2ord, cheby2, sosfilt, sosfiltfilt

if TYPE_CHECKING:
    import mne


EEG_CHANNEL_COUNT = 22
CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")
FILTER_BANK_BANDS = tuple((float(low), float(low + 4)) for low in range(4, 40, 4))


@dataclass(frozen=True)
class PreprocessConfig:
    sfreq: float = 250.0
    epoch_start: float = 2.0
    epoch_end: float = 6.0
    low_freq: float = 4.0
    high_freq: float = 40.0
    filter_order: int = 6
    stop_attenuation_db: float = 60.0

    @property
    def n_times(self) -> int:
        return int(round((self.epoch_end - self.epoch_start) * self.sfreq))


def _load_labels(label_path: Path) -> np.ndarray:
    values = loadmat(label_path)["classlabel"].reshape(-1).astype(np.int64)
    if not np.isin(values, (1, 2, 3, 4)).all():
        raise ValueError(f"Unexpected labels in {label_path}")
    return values - 1


def _artifact_mask(raw: mne.io.BaseRaw, trial_onsets: np.ndarray) -> np.ndarray:
    rejected = np.asarray(
        [ann["onset"] for ann in raw.annotations if ann["description"] == "1023"],
        dtype=np.float64,
    )
    if rejected.size == 0:
        return np.zeros(trial_onsets.size, dtype=bool)
    next_onsets = np.r_[trial_onsets[1:], raw.times[-1] + 1.0]
    return np.asarray(
        [np.any((rejected >= start) & (rejected < stop)) for start, stop in zip(trial_onsets, next_onsets)],
        dtype=bool,
    )


def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float,
              order: int, attenuation: float) -> np.ndarray:
    sos = cheby2(
        order,
        attenuation,
        (low, high),
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)


def read_gdf_session(
    gdf_path: str | Path,
    label_path: str | Path,
    config: PreprocessConfig = PreprocessConfig(),
) -> dict[str, np.ndarray]:
    """Load one GDF session and return filtered trials plus metadata arrays."""
    # MNE pulls in Numba/LLVM and is expensive to import.  Keep it out of the
    # training and synthetic-test import path; only real GDF ingestion needs it.
    import mne

    gdf_path = Path(gdf_path)
    label_path = Path(label_path)
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, config.sfreq):
        raise ValueError(f"Expected {config.sfreq} Hz, got {sfreq} Hz in {gdf_path}")
    if len(raw.ch_names) < EEG_CHANNEL_COUNT:
        raise ValueError(f"Expected at least 22 EEG channels in {gdf_path}")

    trial_onsets = np.asarray(
        [ann["onset"] for ann in raw.annotations if ann["description"] == "768"],
        dtype=np.float64,
    )
    labels = _load_labels(label_path)
    if trial_onsets.size != labels.size:
        raise ValueError(
            f"Trial/label mismatch for {gdf_path}: {trial_onsets.size} vs {labels.size}"
        )

    eeg = raw.get_data(picks=np.arange(EEG_CHANNEL_COUNT)) * 1e6  # volts -> microvolts
    raw_trials = np.empty((labels.size, EEG_CHANNEL_COUNT, config.n_times), dtype=np.float32)
    offset = int(round(config.epoch_start * sfreq))
    for index, onset in enumerate(trial_onsets):
        start = int(round(onset * sfreq)) + offset
        stop = start + config.n_times
        if stop > eeg.shape[1]:
            raise ValueError(f"Epoch {index} exceeds recording length in {gdf_path}")
        raw_trials[index] = np.nan_to_num(eeg[:, start:stop], copy=False)

    trials = _bandpass(
        raw_trials,
        sfreq,
        config.low_freq,
        config.high_freq,
        config.filter_order,
        config.stop_attenuation_db,
    )
    return {
        "x": trials,
        # Preserve the hardware-filtered signal for FBCNet. Its official
        # pipeline constructs the nine sub-bands directly from raw trials.
        "raw_x": raw_trials,
        "y": labels,
        "artifact": _artifact_mask(raw, trial_onsets),
        "channel_names": np.asarray(raw.ch_names[:EEG_CHANNEL_COUNT]),
        "sfreq": np.asarray(sfreq, dtype=np.float32),
    }


def design_filter_bank(
    sfreq: float = 250.0,
    bands: Iterable[tuple[float, float]] = FILTER_BANK_BANDS,
    passband_ripple_db: float = 3.0,
    attenuation: float = 30.0,
    transition_hz: float = 2.0,
) -> list[np.ndarray]:
    """Design FBCNet's nine Chebyshev-II band-pass filters as SOS sections.

    Exposed separately from ``make_filter_bank`` so streaming inference can
    precompute the filters once and apply them causally per window.
    """
    sos_list: list[np.ndarray] = []
    for low, high in bands:
        passband = (low, high)
        stopband = (low - transition_hz, high + transition_hz)
        order, critical = cheb2ord(
            passband,
            stopband,
            passband_ripple_db,
            attenuation,
            fs=sfreq,
        )
        sos_list.append(
            cheby2(order, attenuation, critical, btype="bandpass", fs=sfreq, output="sos")
        )
    return sos_list


def make_filter_bank(
    trials: np.ndarray,
    sfreq: float = 250.0,
    bands: Iterable[tuple[float, float]] = FILTER_BANK_BANDS,
    passband_ripple_db: float = 3.0,
    attenuation: float = 30.0,
    transition_hz: float = 2.0,
    zero_phase: bool = False,
) -> np.ndarray:
    """Create FBCNet's 9-view representation: trials x bands x channels x time.

    The Chebyshev-II design and 2 Hz transition band follow the authors'
    public ``filterBank`` transform. Their default is causal filtering;
    ``zero_phase=True`` is available for offline ablations.
    """
    sos_list = design_filter_bank(sfreq, bands, passband_ripple_db, attenuation, transition_hz)
    output = np.empty((trials.shape[0], len(sos_list), *trials.shape[1:]), dtype=np.float32)
    for index, sos in enumerate(sos_list):
        filter_fn = sosfiltfilt if zero_phase else sosfilt
        output[:, index] = filter_fn(sos, trials, axis=-1).astype(np.float32, copy=False)
    return output


def preprocess_dataset(
    raw_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    subjects: Iterable[int] = range(1, 10),
    config: PreprocessConfig = PreprocessConfig(),
    overwrite: bool = False,
) -> list[Path]:
    raw_dir, label_dir, output_dir = Path(raw_dir), Path(label_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for subject in subjects:
        for session in ("T", "E"):
            stem = f"A{subject:02d}{session}"
            destination = output_dir / f"{stem}.npz"
            if destination.exists() and not overwrite:
                written.append(destination)
                continue
            result = read_gdf_session(raw_dir / f"{stem}.gdf", label_dir / f"{stem}.mat", config)
            np.savez_compressed(destination, **result)
            written.append(destination)

    metadata = {
        "dataset": "BCI Competition IV 2a",
        "classes": list(CLASS_NAMES),
        "filter_bank_bands_hz": [list(band) for band in FILTER_BANK_BANDS],
        "config": asdict(config),
        "files": [path.name for path in written],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return written


def load_processed_session(
    processed_dir: str | Path,
    subject: int,
    session: str,
    file_prefix: str = "A",
) -> dict[str, np.ndarray]:
    path = Path(processed_dir) / f"{file_prefix}{subject:02d}{session.upper()}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed session: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}

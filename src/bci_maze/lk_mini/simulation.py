"""Deterministic motor-imagery recordings for end-to-end workflow tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .recording import TrialRecord, save_mi_recording
from .source import DEFAULT_CHANNEL_NAMES


SUPPORTED_CLASSES = ("left_hand", "right_hand")


def make_synthetic_mi_epoch(
    label: str,
    *,
    sample_rate: int = 250,
    epoch_seconds: float = 4.0,
    n_channels: int = len(DEFAULT_CHANNEL_NAMES),
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an EEG-like epoch with class-specific sensorimotor covariance."""
    if label not in SUPPORTED_CLASSES:
        raise ValueError(f"Synthetic MI supports {SUPPORTED_CLASSES}, got {label!r}")
    samples = int(round(sample_rate * epoch_seconds))
    if sample_rate <= 0 or samples < 32 or n_channels < 2:
        raise ValueError("Synthetic MI requires at least two channels and 32 samples")

    seconds = np.arange(samples, dtype=np.float64) / sample_rate
    epoch = rng.normal(0.0, 2.5, (n_channels, samples))
    shared_phase = rng.uniform(0.0, 2.0 * np.pi)
    shared = 2.5 * np.sin(2.0 * np.pi * 10.5 * seconds + shared_phase)
    epoch += shared[None, :] * rng.uniform(0.25, 0.55, (n_channels, 1))

    # Deliberately artificial covariance, independent of electrode anatomy.
    # The upper bank must carry class information in the 16-channel workflow.
    dominant = (0, n_channels // 2) if label == "left_hand" else (1, n_channels - 1)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    mu = np.sin(2.0 * np.pi * 10.0 * seconds + phase)
    beta = np.sin(2.0 * np.pi * 18.0 * seconds + phase * 0.5)
    epoch[dominant[0]] += 13.0 * mu + 5.0 * beta
    epoch[dominant[1]] += 10.0 * mu + 4.0 * beta
    return epoch.astype(np.float32)


def save_synthetic_mi_recording(
    path: str | Path,
    *,
    schedule: Sequence[tuple[int, str]],
    subject_id: str,
    session_id: str,
    channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES,
    sample_rate: int = 250,
    epoch_seconds: float = 4.0,
    seed: int = 42,
    configuration: dict[str, object] | None = None,
) -> Path:
    """Create a labeled recording that follows the real NPZ storage contract."""
    if not schedule:
        raise ValueError("Synthetic MI schedule must not be empty")
    names = tuple(channel_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("channel_names must be non-empty and unique")
    rng = np.random.default_rng(seed)
    epochs: list[np.ndarray] = []
    trials: list[TrialRecord] = []
    labels: list[str] = []
    cursor = 0

    for trial_index, (block, label) in enumerate(schedule):
        epoch = make_synthetic_mi_epoch(
            label,
            sample_rate=sample_rate,
            epoch_seconds=epoch_seconds,
            n_channels=len(names),
            rng=rng,
        )
        stop = cursor + epoch.shape[1]
        epochs.append(epoch)
        labels.append(label)
        trials.append(
            TrialRecord(
                trial_index=trial_index,
                block=int(block),
                label=label,
                start_sample=cursor,
                stop_sample=stop,
                cue_monotonic=cursor / sample_rate,
                artifact=False,
            )
        )
        cursor = stop

    stacked = np.stack(epochs)
    continuous = np.concatenate(epochs, axis=1)
    timestamps = np.arange(cursor, dtype=np.float64) / sample_rate
    return save_mi_recording(
        path,
        continuous=continuous,
        timestamps=timestamps,
        epochs=stacked,
        labels=labels,
        trials=trials,
        sample_rate=sample_rate,
        channel_names=names,
        subject_id=subject_id,
        session_id=session_id,
        configuration={
            **(configuration or {}),
            "backend": "synthetic_mi",
            "label_source": "scripted_simulation",
            "reference": "synthetic",
            "reference_location": "not_applicable",
            "bias_location": "not_applicable",
            "epoch_seconds": epoch_seconds,
            "seed": seed,
            "purpose": "automated workflow validation; not human EEG",
        },
    )

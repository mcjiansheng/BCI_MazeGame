"""Thread-safe continuous recorder and MI recording file helpers."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .source import EEGSource


@dataclass
class TrialRecord:
    trial_index: int
    block: int
    label: str
    start_sample: int
    stop_sample: int
    cue_monotonic: float
    artifact: bool = False


class ContinuousRecorder:
    """Drain an EEGSource on a worker thread without blocking the UI."""

    def __init__(self, source: EEGSource):
        self.source = source
        self._chunks: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._samples = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._samples

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Recorder already started")
        self.source.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="EEG recorder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                chunk = self.source.read(timeout=0.25)
                if not chunk.data.shape[1]:
                    continue
                with self._lock:
                    self._chunks.append(chunk.data.copy())
                    self._timestamps.append(chunk.monotonic_timestamps.copy())
                    self._samples += chunk.data.shape[1]
        except BaseException as exception:
            self.error = exception
            self._stop_event.set()

    def wait_for_samples(self, stop_sample: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error:
                raise RuntimeError("EEG recorder stopped unexpectedly") from self.error
            if self.sample_count >= stop_sample:
                return True
            time.sleep(0.01)
        return self.sample_count >= stop_sample

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if not self._chunks:
                return (
                    np.empty((len(self.source.channel_names), 0), dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )
            return np.concatenate(self._chunks, axis=1), np.concatenate(self._timestamps)

    def recent(self, samples: int) -> np.ndarray:
        if samples <= 0:
            raise ValueError("samples must be positive")
        with self._lock:
            remaining = min(samples, self._samples)
            selected: list[np.ndarray] = []
            for chunk in reversed(self._chunks):
                if remaining <= 0:
                    break
                take = min(remaining, chunk.shape[1])
                selected.append(chunk[:, -take:])
                remaining -= take
            if not selected:
                return np.empty((len(self.source.channel_names), 0), dtype=np.float64)
            return np.concatenate(selected[::-1], axis=1)

    def segment(self, start: int, stop: int) -> np.ndarray:
        with self._lock:
            if start < 0 or stop > self._samples or stop <= start:
                raise ValueError(f"Invalid segment [{start}:{stop}] for {self._samples} samples")
            pieces: list[np.ndarray] = []
            cursor = 0
            for chunk in self._chunks:
                chunk_stop = cursor + chunk.shape[1]
                if chunk_stop <= start:
                    cursor = chunk_stop
                    continue
                if cursor >= stop:
                    break
                local_start = max(0, start - cursor)
                local_stop = min(chunk.shape[1], stop - cursor)
                pieces.append(chunk[:, local_start:local_stop])
                cursor = chunk_stop
            result = np.concatenate(pieces, axis=1)
            if result.shape[1] != stop - start:
                raise RuntimeError("Recorder segment length mismatch")
            return result

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.source.stop()

    def __enter__(self) -> "ContinuousRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def save_mi_recording(
    path: str | Path,
    *,
    continuous: np.ndarray,
    timestamps: np.ndarray,
    epochs: np.ndarray,
    labels: Sequence[str],
    trials: Sequence[TrialRecord],
    sample_rate: int,
    channel_names: Sequence[str],
    subject_id: str,
    session_id: str,
    configuration: dict[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "subject_id": subject_id,
        "session_id": session_id,
        "sample_rate": sample_rate,
        "channel_names": list(channel_names),
        "configuration": configuration,
        "trials": [asdict(trial) for trial in trials],
    }
    np.savez_compressed(
        destination,
        continuous=np.asarray(continuous, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        epochs=np.asarray(epochs, dtype=np.float32),
        labels=np.asarray(labels),
        artifact=np.asarray([trial.artifact for trial in trials], dtype=bool),
        blocks=np.asarray([trial.block for trial in trials], dtype=np.int16),
        sample_rate=np.asarray(sample_rate, dtype=np.int32),
        channel_names=np.asarray(channel_names),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return destination


def load_mi_recordings(paths: Sequence[str | Path]) -> dict[str, np.ndarray | list[dict]]:
    all_epochs, all_labels, all_artifact, all_blocks = [], [], [], []
    expected_rate: int | None = None
    expected_channels: tuple[str, ...] | None = None
    metadata: list[dict] = []
    block_offset = 0
    for raw_path in paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as loaded:
            rate = int(np.asarray(loaded["sample_rate"]).item())
            channels = tuple(str(value) for value in loaded["channel_names"].tolist())
            if expected_rate is None:
                expected_rate, expected_channels = rate, channels
            if rate != expected_rate or channels != expected_channels:
                raise ValueError(f"Recording {path} has incompatible sample rate or channel order")
            epochs = np.asarray(loaded["epochs"], dtype=np.float64)
            labels = np.asarray(loaded["labels"]).astype(str)
            artifact = np.asarray(loaded["artifact"], dtype=bool)
            blocks = np.asarray(loaded["blocks"], dtype=np.int64) + block_offset
            if not (len(epochs) == len(labels) == len(artifact) == len(blocks)):
                raise ValueError(f"Recording {path} has inconsistent trial arrays")
            all_epochs.append(epochs)
            all_labels.append(labels)
            all_artifact.append(artifact)
            all_blocks.append(blocks)
            metadata.append(json.loads(str(np.asarray(loaded["metadata_json"]).item())))
            block_offset = int(blocks.max(initial=block_offset)) + 1
    if not all_epochs:
        raise ValueError("At least one recording is required")
    return {
        "epochs": np.concatenate(all_epochs),
        "labels": np.concatenate(all_labels),
        "artifact": np.concatenate(all_artifact),
        "blocks": np.concatenate(all_blocks),
        "sample_rate": np.asarray(expected_rate),
        "channel_names": np.asarray(expected_channels),
        "metadata": metadata,
    }

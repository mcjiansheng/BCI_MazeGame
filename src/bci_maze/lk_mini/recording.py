"""Thread-safe continuous recorder and MI recording file helpers."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .source import EEGSource, _SourceAlreadyStarted
from .timing import StreamTimingGuard
from .provenance import BACKEND_SOURCES
from bci_maze.stopping import defer_stop


@dataclass
class TrialRecord:
    trial_index: int
    block: int
    label: str
    start_sample: int
    stop_sample: int
    cue_monotonic: float
    artifact: bool = False
    segment_id: int = 0


class ContinuousRecorder:
    """Drain an EEGSource on a worker thread without blocking the UI."""

    def __init__(self, source: EEGSource):
        self.source = source
        self._chunks: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._received_timestamps: list[np.ndarray] = []
        self._segment_ids: list[int] = []
        self._segment_id = 0
        self._samples = 0
        self._last_received_at: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._samples

    @property
    def segment_id(self) -> int:
        with self._lock:
            return self._segment_id

    @property
    def recovering(self) -> bool:
        return bool(getattr(self.source, "recovering", False))

    @property
    def last_sample_age(self) -> float | None:
        """Wall-clock age of the newest batch received by this recorder."""
        with self._lock:
            received = self._last_received_at
        return None if received is None else max(0.0, time.monotonic() - received)

    def cursor(self) -> tuple[int, int]:
        """Take an atomic sample/segment marker at the imagery cue."""
        with self._lock:
            return self._samples, self._segment_id

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Recorder already started")
        try:
            self.source.start()
            with defer_stop():
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._run, name="EEG recorder", daemon=True)
                self._thread.start()
        except _SourceAlreadyStarted:
            raise
        except BaseException:
            try:
                self.stop()
            except BaseException:
                pass  # Preserve the start failure, including cancellation.
            raise

    def _run(self) -> None:
        try:
            timing = None if getattr(self.source, "validates_timing", False) else StreamTimingGuard(self.source.sample_rate)
            last_sample_time = time.monotonic()
            while not self._stop_event.is_set():
                chunk = self.source.read(timeout=0.25)
                chunk_segment = int(chunk.segment_id)
                with self._lock:
                    changed = chunk_segment != self._segment_id
                    self._segment_id = chunk_segment
                if changed or self.recovering:
                    if timing is not None:
                        timing = StreamTimingGuard(self.source.sample_rate)
                    last_sample_time = time.monotonic()
                if not chunk.data.shape[1]:
                    if self.source.exhausted or (
                        not self.recovering and time.monotonic() - last_sample_time >= 5.0
                    ):
                        raise RuntimeError("EEG source ended or produced no samples for 5 seconds")
                    continue
                if timing is not None:
                    timing.check(chunk.monotonic_timestamps)
                last_sample_time = time.monotonic()
                with self._lock:
                    self._chunks.append(chunk.data.copy())
                    self._timestamps.append(chunk.monotonic_timestamps.copy())
                    self._received_timestamps.append(
                        chunk.received_timestamps.copy() if chunk.received_timestamps is not None
                        else np.full(chunk.data.shape[1], np.nan)
                    )
                    self._segment_ids.append(chunk_segment)
                    self._samples += chunk.data.shape[1]
                    self._last_received_at = last_sample_time
        except BaseException as exception:
            self.error = exception
            self._stop_event.set()

    def wait_for_samples(
        self, stop_sample: int, timeout: float, *, segment_id: int | None = None
    ) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            # A later interruption does not invalidate samples already received.
            # Check the actual endpoint's segment, not the current live segment;
            # segment(start, stop) still validates the entire requested window.
            available = self._target_available(stop_sample, segment_id)
            if available is not None:
                return available
            if self.error:
                raise RuntimeError("EEG recorder stopped unexpectedly") from self.error
            if segment_id is not None and (self.segment_id != segment_id or self.recovering):
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def _target_available(self, stop: int, segment: int | None) -> bool | None:
        with self._lock:
            if stop > self._samples:
                return None
            if segment is None or stop <= 0:
                return True
            cursor = self._samples
            for chunk, chunk_segment in zip(reversed(self._chunks), reversed(self._segment_ids)):
                start = cursor - chunk.shape[1]
                if start < stop <= cursor:
                    return chunk_segment == segment
                cursor = start
        return False

    def snapshot_segment_ids(self) -> np.ndarray:
        with self._lock:
            return np.concatenate([
                np.full(chunk.shape[1], segment, dtype=np.int64)
                for chunk, segment in zip(self._chunks, self._segment_ids)
            ]) if self._chunks else np.empty(0, dtype=np.int64)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if not self._chunks:
                return (
                    np.empty((len(self.source.channel_names), 0), dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )
            return np.concatenate(self._chunks, axis=1), np.concatenate(self._timestamps)

    def snapshot_received_timestamps(self) -> np.ndarray:
        """Raw source timestamps; NaN marks sources without reception timestamps."""
        with self._lock:
            return np.concatenate(self._received_timestamps) if self._received_timestamps else np.empty(0)

    def recent(self, samples: int) -> np.ndarray:
        if samples <= 0:
            raise ValueError("samples must be positive")
        with self._lock:
            remaining = min(samples, self._samples)
            selected: list[np.ndarray] = []
            for chunk, segment in zip(reversed(self._chunks), reversed(self._segment_ids)):
                if segment != self._segment_id:
                    break
                if remaining <= 0:
                    break
                take = min(remaining, chunk.shape[1])
                selected.append(chunk[:, -take:])
                remaining -= take
            if not selected:
                return np.empty((len(self.source.channel_names), 0), dtype=np.float64)
            return np.concatenate(selected[::-1], axis=1)

    def segment(self, start: int, stop: int, *, segment_id: int | None = None) -> np.ndarray:
        with self._lock:
            if start < 0 or stop > self._samples or stop <= start:
                raise ValueError(f"Invalid segment [{start}:{stop}] for {self._samples} samples")
            pieces: list[np.ndarray] = []
            selected_segments: set[int] = set()
            cursor = 0
            for chunk, chunk_segment in zip(self._chunks, self._segment_ids):
                chunk_stop = cursor + chunk.shape[1]
                if chunk_stop <= start:
                    cursor = chunk_stop
                    continue
                if cursor >= stop:
                    break
                local_start = max(0, start - cursor)
                local_stop = min(chunk.shape[1], stop - cursor)
                pieces.append(chunk[:, local_start:local_stop])
                selected_segments.add(chunk_segment)
                cursor = chunk_stop
            if len(selected_segments) != 1 or (
                segment_id is not None and selected_segments != {segment_id}
            ):
                raise ValueError("Trial crosses an EEG stream segment boundary")
            result = np.concatenate(pieces, axis=1)
            if result.shape[1] != stop - start:
                raise RuntimeError("Recorder segment length mismatch")
            return result

    @defer_stop()
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.source.stop()

    def __enter__(self) -> "ContinuousRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


@defer_stop()
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
    segment_ids: np.ndarray | None = None,
    discarded_trials: Sequence[dict[str, object]] = (),
    recovery_stats: dict[str, object] | None = None,
    recovery_events: Sequence[dict[str, object]] = (),
    received_timestamps: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
    trial_summary_override: dict[str, object] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backend = str(configuration.get("backend", "unspecified"))
    source_kind = BACKEND_SOURCES.get(backend, "unspecified")
    provenance = {
        "kind": source_kind,
        "source_kind": source_kind,
        "backend": backend,
        "label_source": configuration.get(
            "label_source", "scripted_simulation" if backend == "synthetic_mi" else "unspecified"
        ),
        "human_mi_verified": False,
    }
    if source_kind == "replay":
        provenance["source_recording"] = configuration.get("source_recording", "unspecified")
        provenance["source_provenance"] = configuration.get("source_provenance", {})
    completed = len(trials)
    artifacts = sum(bool(trial.artifact) for trial in trials)
    planned = int(configuration.get("planned_trials", completed + len(discarded_trials)))
    metadata = {
        "format_version": 3,
        "subject_id": subject_id,
        "session_id": session_id,
        "sample_rate": sample_rate,
        "channel_names": list(channel_names),
        "units": "uV",
        "reference": configuration.get("reference", "unspecified"),
        "reference_location": configuration.get("reference_location", "unspecified"),
        "bias_location": configuration.get("bias_location", "unspecified"),
        "provenance": provenance,
        "trial_summary": {
            "planned": planned,
            "completed": completed,
            "usable": completed - artifacts,
            "artifact_marked": artifacts,
            "discarded": len(discarded_trials),
            "unattempted": max(0, planned - completed - len(discarded_trials)),
            "usable_definition": "completed trials without an operator artifact mark",
        },
        "configuration": configuration,
        "trials": [asdict(trial) for trial in trials],
        "discarded_trials": list(discarded_trials),
        "recovery_stats": recovery_stats or {},
        "recovery_events": list(recovery_events),
    }
    segment_ids = np.zeros(np.asarray(timestamps).size, dtype=np.int64) if segment_ids is None else np.asarray(segment_ids, dtype=np.int64)
    if segment_ids.shape != np.asarray(timestamps).shape:
        raise ValueError("One segment ID is required per continuous EEG sample")
    metadata["trial_summary"].update(trial_summary_override or {})
    arrays = dict(
        continuous=np.asarray(continuous, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        segment_ids=segment_ids,
        epochs=np.asarray(epochs, dtype=np.float32),
        labels=np.asarray(labels),
        artifact=np.asarray([trial.artifact for trial in trials], dtype=bool),
        blocks=np.asarray([trial.block for trial in trials], dtype=np.int16),
        sample_rate=np.asarray(sample_rate, dtype=np.int32),
        channel_names=np.asarray(channel_names),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    if received_timestamps is not None:
        received = np.asarray(received_timestamps, dtype=np.float64)
        if received.shape != np.asarray(timestamps).shape:
            raise ValueError("One received timestamp is required per continuous EEG sample")
        arrays["received_timestamps"] = received
    if extra_arrays:
        overlap = arrays.keys() & extra_arrays.keys()
        if overlap:
            raise ValueError(f"Extra arrays overwrite recording fields: {sorted(overlap)}")
        arrays.update(extra_arrays)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".npz", dir=destination.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def load_mi_recordings(paths: Sequence[str | Path]) -> dict[str, np.ndarray | list[dict]]:
    paths = [Path(path).resolve() for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Provide unique recording paths; the same recording was selected more than once")
    all_epochs, all_labels, all_artifact, all_blocks = [], [], [], []
    expected_rate: int | None = None
    expected_channels: tuple[str, ...] | None = None
    metadata: list[dict] = []
    block_offset = 0
    for raw_path in paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as loaded:
            rate_value = np.asarray(loaded["sample_rate"]).item()
            if not np.isfinite(rate_value) or rate_value <= 0 or rate_value != int(rate_value):
                raise ValueError(f"Recording {path} requires a positive integer sample rate")
            rate = int(rate_value)
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

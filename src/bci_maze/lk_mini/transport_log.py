"""Optional read-level EEG transport evidence, separate from decision logging."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class TransportLogger:
    def __init__(self, path: Path | None):
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", encoding="utf-8")
        self.started = time.monotonic()
        self.sample_rate = 250
        self.samples = 0
        self.segments: dict[int, dict] = {}
        self.state_key = None
        self.previous_recovering = False
        self.pending_recovery = False
        self.first_recovery = None
        self.last_read = None
        self.nonfinite_data_values = 0
        self.nonfinite_timestamps = 0

    def write(self, event: str, now: float | None = None, **values) -> None:
        if self.file is None:
            return
        now = time.monotonic() if now is None else now
        record = {"event": event, "monotonic": now,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed": now - self.started, **values}
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def start(self, started: float, sample_rate: int, channel_names, window_samples: int) -> None:
        self.started = started
        self.sample_rate = sample_rate
        self.write("start", started, sample_rate=sample_rate, channel_names=list(channel_names),
                   window_samples=window_samples, timestamp_domain="monotonic_seconds")

    def observe(self, chunk, source, now: float) -> None:
        segment = int(chunk.segment_id)
        count = int(chunk.data.shape[1])
        if self.file is None:
            # Keep the decision's sample count without inspecting transport state.
            entry = self.segments.setdefault(segment, {"sample_count": 0})
            entry["sample_count"] += count
            return
        recovering = bool(getattr(source, "recovering", False))
        state = str(getattr(source, "connection_state", "streaming"))
        events = list(getattr(source, "recovery_events", []))
        key = (state, recovering, segment, len(events))
        first_recovery = False
        if recovering and not self.previous_recovering and self.samples:
            self.pending_recovery = True
            if self.first_recovery is None:
                self.first_recovery = now
                first_recovery = True
        if key != self.state_key:
            self.write("state", now, state=state, recovering=recovering, segment_id=segment,
                       first_recovery=first_recovery,
                       recovery_stats=dict(getattr(source, "recovery_stats", {})), recovery_events=events)
            self.state_key = key
        self.previous_recovering = recovering
        if not count:
            return
        timestamps = np.asarray(chunk.monotonic_timestamps, dtype=np.float64)
        nonfinite_data = int(np.count_nonzero(~np.isfinite(chunk.data)))
        nonfinite_times = int(np.count_nonzero(~np.isfinite(timestamps)))
        self.nonfinite_data_values += nonfinite_data
        self.nonfinite_timestamps += nonfinite_times
        entry = self.segments.get(segment)
        first = entry is None
        if first:
            entry = {"segment_id": segment, "sample_count": 0, "chunk_count": 0,
                     "first_read_monotonic": now, "first_read_elapsed": now - self.started,
                     "first_sample_timestamp": float(timestamps[0]), "continuity_errors": 0,
                     "max_read_gap_seconds": 0.0}
            self.segments[segment] = entry
        previous_timestamp = entry.get("last_sample_timestamp")
        joined = timestamps if previous_timestamp is None else np.r_[previous_timestamp, timestamps]
        intervals = np.diff(joined)
        invalid = int(np.count_nonzero(~np.isfinite(intervals) | (intervals <= 0) | (intervals > 2.5 / self.sample_rate)))
        if "last_read_monotonic" in entry:
            entry["max_read_gap_seconds"] = max(entry["max_read_gap_seconds"], now - entry["last_read_monotonic"])
        entry.update(last_read_monotonic=now, last_read_elapsed=now - self.started,
                     last_sample_timestamp=float(timestamps[-1]))
        entry["sample_count"] += count
        entry["chunk_count"] += 1
        entry["continuity_errors"] += invalid
        self.samples += count
        received = chunk.received_timestamps
        self.write("chunk", now, state=state, recovering=recovering, segment_id=segment,
                   sample_count=count, total_samples=self.samples,
                   segment_samples=entry["sample_count"], first_in_segment=first,
                   first_after_recovery=self.pending_recovery,
                   timestamp_start=float(timestamps[0]), timestamp_end=float(timestamps[-1]),
                   received_timestamp_start=None if received is None else float(received[0]),
                   received_timestamp_end=None if received is None else float(received[-1]),
                   previous_timestamp_gap=None if previous_timestamp is None else float(timestamps[0] - previous_timestamp),
                   continuity_errors=invalid,
                   nonfinite_data_values=nonfinite_data, nonfinite_timestamps=nonfinite_times,
                   max_sample_interval=None if not intervals.size else float(np.max(intervals)),
                   min_sample_interval=None if not intervals.size else float(np.min(intervals)),
                   read_gap_seconds=None if self.last_read is None else now - self.last_read)
        self.last_read = now
        self.pending_recovery = False

    def finish(self, source, *, capture_end: float | None, error: str = "") -> None:
        if self.file is None:
            return
        now = time.monotonic()
        capture_end = now if capture_end is None else capture_end
        try:
            self.write("summary", now, capture_end_monotonic=capture_end,
                       acquisition_elapsed=capture_end - self.started,
                       samples=self.samples, segments=list(self.segments.values()),
                       nonfinite_data_values=self.nonfinite_data_values, nonfinite_timestamps=self.nonfinite_timestamps,
                       first_recovery_elapsed=None if self.first_recovery is None else self.first_recovery - self.started,
                       last_data_elapsed=None if self.last_read is None else self.last_read - self.started,
                       tail_silence_seconds=None if self.last_read is None else max(0.0, capture_end - self.last_read),
                       state=str(getattr(source, "connection_state", "unknown")),
                       recovery_stats=dict(getattr(source, "recovery_stats", {})),
                       recovery_events=list(getattr(source, "recovery_events", [])), error=error)
        finally:
            self.file.close()
            self.file = None

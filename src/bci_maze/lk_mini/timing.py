"""Shared checks for timestamps supplied by external EEG streams."""

from __future__ import annotations

import numpy as np


def timestamp_intervals(values: np.ndarray, previous: float | None) -> np.ndarray:
    return np.diff(values, prepend=previous) if previous is not None else np.diff(values)


def count_discontinuities(intervals: np.ndarray, max_gap: float | None) -> int:
    invalid = ~np.isfinite(intervals) | (intervals <= 0)
    if max_gap is not None:
        invalid |= intervals > max_gap
    return int(np.count_nonzero(invalid))


class StreamTimingGuard:
    def __init__(self, sample_rate: int | None = None, *, max_gap: float | None = None):
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.max_gap = 2.5 / sample_rate if max_gap is None and sample_rate is not None else max_gap
        self.previous: float | None = None

    def check(self, timestamps: np.ndarray) -> np.ndarray:
        values = np.asarray(timestamps, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("EEG timestamps must be a finite one-dimensional array")
        if not values.size:
            return values
        intervals = timestamp_intervals(values, self.previous)
        if count_discontinuities(intervals, self.max_gap):
            raise ValueError("EEG timestamp discontinuity: duplicate, reversed or missing samples")
        self.previous = float(values[-1])
        return intervals

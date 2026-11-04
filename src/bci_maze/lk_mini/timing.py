"""Reject discontinuous EEG before trial extraction or online classification."""

from __future__ import annotations

import numpy as np


class StreamTimingGuard:
    def __init__(self, sample_rate: int):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.max_gap = 2.5 / sample_rate
        self.previous: float | None = None

    def check(self, timestamps: np.ndarray) -> None:
        values = np.asarray(timestamps, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("EEG timestamps must be a finite one-dimensional array")
        if not values.size:
            return
        joined = values if self.previous is None else np.r_[self.previous, values]
        intervals = np.diff(joined)
        if np.any(intervals <= 0) or np.any(intervals > self.max_gap):
            raise ValueError("EEG timestamp discontinuity: duplicate, reversed or missing samples")
        self.previous = float(values[-1])

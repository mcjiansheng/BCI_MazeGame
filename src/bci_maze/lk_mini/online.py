"""Rolling window and confidence rejection for online MI decisions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class RollingEEGBuffer:
    def __init__(self, n_channels: int, capacity_samples: int):
        self.n_channels = int(n_channels)
        self.capacity_samples = int(capacity_samples)
        self._data = np.empty((self.n_channels, 0), dtype=np.float64)

    @property
    def sample_count(self) -> int:
        return self._data.shape[1]

    def append(self, data: np.ndarray) -> None:
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.n_channels:
            raise ValueError("buffer append requires channels x samples")
        if not values.shape[1]:
            return
        self._data = np.concatenate((self._data, values), axis=1)[:, -self.capacity_samples :]

    def latest(self, samples: int) -> np.ndarray:
        if samples <= 0 or self.sample_count < samples:
            raise ValueError(f"Need {samples} samples; buffer has {self.sample_count}")
        return self._data[:, -samples:].copy()


@dataclass(frozen=True)
class OnlineDecision:
    label: str
    confidence: float
    probabilities: dict[str, float]
    accepted: bool
    reason: str


class ProbabilitySmoother:
    def __init__(
        self,
        classes: Sequence[str],
        *,
        history: int = 3,
        confidence_threshold: float = 0.65,
        margin_threshold: float = 0.15,
    ):
        self.classes = tuple(classes)
        self.history = deque(maxlen=int(history))
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)

    def update(self, probabilities: Sequence[float]) -> OnlineDecision:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (len(self.classes),) or not np.isfinite(values).all():
            raise ValueError("probabilities do not match model classes")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value")
        values = values / total
        self.history.append(values)
        smoothed = np.mean(self.history, axis=0)
        order = np.argsort(smoothed)[::-1]
        best, second = int(order[0]), int(order[1])
        confidence = float(smoothed[best])
        margin = float(smoothed[best] - smoothed[second])
        enough_history = len(self.history) == self.history.maxlen
        if not enough_history:
            accepted, reason = False, "warming_up"
        elif confidence < self.confidence_threshold:
            accepted, reason = False, "low_confidence"
        elif margin < self.margin_threshold:
            accepted, reason = False, "small_margin"
        else:
            accepted, reason = True, "accepted"
        label = self.classes[best] if accepted else "unknown"
        return OnlineDecision(
            label=label,
            confidence=confidence,
            probabilities={name: float(smoothed[index]) for index, name in enumerate(self.classes)},
            accepted=accepted,
            reason=reason,
        )

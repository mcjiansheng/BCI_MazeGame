"""Online decoder: causal preprocessing, window inference, hysteresis policy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.signal import cheby2, sosfilt

from ..models import build_model
from ..preprocessing import design_filter_bank
from ..training import apply_norm_stats, uses_filter_bank, uses_raw_broadband

VALID_COMMANDS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class LabelMap:
    """Maps classifier class indices to maze commands (None = rest/stop)."""

    labels: tuple[str, ...]
    commands: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        if len(self.labels) != len(self.commands):
            raise ValueError("labels and commands must have the same length")
        for command in self.commands:
            if command is not None and command not in VALID_COMMANDS:
                raise ValueError(f"Invalid command {command!r}; choose from {VALID_COMMANDS}")

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def command_for(self, label: int) -> str | None:
        return self.commands[label]


def load_label_map(path: str | Path) -> LabelMap:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = tuple(payload["labels"])
    commands_table = payload["commands"]
    missing = [label for label in labels if label not in commands_table]
    if missing:
        raise ValueError(f"Label map has no command entries for {missing}")
    commands = tuple(commands_table.get(label) for label in labels)
    return LabelMap(labels=labels, commands=commands)


@dataclass
class HysteresisPolicy:
    """Dual-threshold state machine that debounces window-level predictions.

    The policy keeps the current state while the posterior probability stays
    inside the hysteresis band ``[exit_threshold, enter_threshold]``, so small
    probability fluctuations around a threshold do not flip the command. Only
    the highest-probability class may enter or replace the current state.
    """

    enter_threshold: float = 0.6
    exit_threshold: float = 0.4
    state: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.exit_threshold <= self.enter_threshold <= 1.0:
            raise ValueError("Require 0 <= exit_threshold <= enter_threshold <= 1")

    def reset(self) -> None:
        self.state = None

    def update(self, probabilities: np.ndarray) -> int | None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError("probabilities must be a non-empty one-dimensional array")
        if not np.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite")
        if self.state is not None and self.state >= probabilities.size:
            raise ValueError("probability vector no longer contains the active state")
        if self.state is not None and probabilities[self.state] < self.exit_threshold:
            self.state = None
        candidate = int(np.argmax(probabilities))
        top = float(probabilities[candidate])
        if self.state is None:
            if top >= self.enter_threshold:
                self.state = candidate
        elif candidate != self.state and top >= self.enter_threshold:
            self.state = candidate
        return self.state


class CausalBandpass:
    """Streaming 4-40 Hz Chebyshev-II band-pass with persistent filter state.

    Offline training used zero-phase ``sosfiltfilt``; that is not realizable
    online, so the streaming path applies the causal forward pass with state
    carried across ``push`` calls.
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float = 250.0,
        low_freq: float = 4.0,
        high_freq: float = 40.0,
        order: int = 6,
        stop_attenuation_db: float = 60.0,
    ):
        self.n_channels = n_channels
        self.sos = cheby2(
            order,
            stop_attenuation_db,
            (low_freq, high_freq),
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        self.reset()

    def reset(self) -> None:
        n_sections = self.sos.shape[0]
        # scipy expects zi with shape (n_sections, *other_dims, 2) for the
        # chosen axis; here the input layout is (n_channels, n_samples).
        self._zi = np.zeros((n_sections, self.n_channels, 2), dtype=np.float64)

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Filter new samples; input shape ``(n_channels,)`` or ``(n_channels, n)``."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {samples.shape[0]}")
        filtered, self._zi = sosfilt(self.sos, samples, axis=-1, zi=self._zi)
        return filtered.astype(np.float32)


class OnlineDecoder:
    """Checkpoint-backed streaming decoder.

    Pipeline per decoded window:

    1. take the most recent ``n_times`` raw samples;
    2. build the model-specific view (nine causal filter-bank views directly
       from raw samples for FBC models, raw input for raw-broadband models, or
       a causal 4--40 Hz view for the remaining broadband models);
    3. normalize with the statistics frozen at training time;
    4. run the model and turn probabilities into a state via the hysteresis
       policy; the state is mapped to a maze command (or None for rest).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        label_map: LabelMap,
        n_channels: int = 22,
        n_times: int = 1000,
        sfreq: float = 250.0,
        device: str | torch.device | None = None,
        policy: HysteresisPolicy | None = None,
        norm_stats: dict | None = None,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model_name = str(checkpoint["model_name"])
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model(
            self.model_name,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=label_map.n_classes,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()

        stats = norm_stats or checkpoint.get("norm_stats")
        if stats is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} carries no norm_stats; retrain with the "
                "current pipeline or pass norm_stats explicitly."
            )
        self.norm_stats = stats
        self.filter_bank_input = uses_filter_bank(self.model_name)
        self.raw_broadband_input = uses_raw_broadband(self.model_name)
        expected_mode = "filter_bank" if self.filter_bank_input else "broadband"
        if stats["mode"] != expected_mode:
            raise ValueError(
                f"norm_stats mode {stats['mode']!r} does not match model input {expected_mode!r}"
            )

        self.label_map = label_map
        self.policy = policy or HysteresisPolicy()
        self.n_channels = n_channels
        self.n_times = n_times
        self.sfreq = sfreq
        self.bandpass = (
            None
            if self.filter_bank_input or self.raw_broadband_input
            else CausalBandpass(n_channels, sfreq=sfreq)
        )
        self._buffer = np.zeros((n_channels, 0), dtype=np.float32)
        if self.filter_bank_input:
            self._bank_sos = design_filter_bank(sfreq=sfreq)

    # ------------------------------------------------------------------ input
    def push_raw(self, samples: np.ndarray) -> None:
        """Push raw microvolt samples and keep only the trailing window."""
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected ({self.n_channels}, n_samples), got {values.shape}"
            )
        prepared = self.bandpass.push(values) if self.bandpass is not None else values
        self._buffer = np.concatenate([self._buffer, prepared], axis=1)
        if self._buffer.shape[1] > self.n_times:
            self._buffer = self._buffer[:, -self.n_times :]

    def reset(self) -> None:
        self.reset_window()
        self.policy.reset()

    def reset_window(self) -> None:
        """Clear buffered samples and filter state, keep the decision policy."""
        self._buffer = np.zeros((self.n_channels, 0), dtype=np.float32)
        if self.bandpass is not None:
            self.bandpass.reset()

    @property
    def ready(self) -> bool:
        return self._buffer.shape[1] >= self.n_times

    # ----------------------------------------------------------------- decode
    def _build_window(self) -> np.ndarray:
        trial = self._buffer[:, -self.n_times :].astype(np.float32, copy=True)
        if self.filter_bank_input:
            views = np.empty((len(self._bank_sos), self.n_channels, self.n_times), dtype=np.float32)
            for index, sos in enumerate(self._bank_sos):
                # Causal per-window filtering mirrors training, where each
                # trial was filtered independently from a zero filter state.
                views[index] = sosfilt(sos, trial, axis=-1).astype(np.float32)
            return apply_norm_stats(views, self.norm_stats)
        return apply_norm_stats(trial[None], self.norm_stats)[None]

    @torch.no_grad()
    def decode_window(self) -> dict:
        """Decode the trailing window; returns probabilities, state and command."""
        if not self.ready:
            raise RuntimeError(
                f"Window not ready: {self._buffer.shape[1]}/{self.n_times} samples buffered"
            )
        window = self._build_window()
        logits = self.model(torch.from_numpy(window).to(self.device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float64)
        state = self.policy.update(probabilities)
        label = int(np.argmax(probabilities))
        command = self.label_map.command_for(state) if state is not None else None
        return {
            "label": label,
            "label_name": self.label_map.labels[label],
            "state": state,
            "state_name": self.label_map.labels[state] if state is not None else None,
            "command": command,
            "confidence": float(probabilities[label]),
            "probabilities": probabilities,
        }

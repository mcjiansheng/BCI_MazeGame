"""EEG data sources for LK-Mini, OpenBCI GUI LSL, replay and tests.

Hardware-specific packages are imported lazily. This lets preprocessing,
training and automated tests run on machines without BrainFlow or pylsl.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_CHANNEL_NAMES = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4")
CHANNEL_CODES = "12345678QWERTYUI"
RATE_COMMANDS = {250: "~6", 500: "~5", 1000: "~4"}
GAIN_CODES = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4, 12: 5, 24: 6}


@dataclass(frozen=True)
class EEGChunk:
    """One chunk of EEG in microvolts, shaped channels x samples."""

    data: np.ndarray
    monotonic_timestamps: np.ndarray

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        timestamps = np.asarray(self.monotonic_timestamps, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("EEG data must be channels x samples")
        if timestamps.ndim != 1 or timestamps.size != data.shape[1]:
            raise ValueError("One timestamp is required per EEG sample")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "monotonic_timestamps", timestamps)


class EEGSource(ABC):
    """Common interface used by device check, collection and online inference."""

    sample_rate: int
    channel_names: tuple[str, ...]

    @abstractmethod
    def start(self) -> None:
        """Open the source and start streaming."""

    @abstractmethod
    def read(self, timeout: float = 0.25) -> EEGChunk:
        """Return newly available data or an empty chunk."""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release all resources."""

    def __enter__(self) -> "EEGSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def build_channel_command(
    channel_index: int,
    gain: int = 24,
    *,
    power_down: bool = False,
    bias: bool = True,
    srb2: bool = True,
    srb1: bool = False,
) -> str:
    """Build the Cyton+Daisy channel command documented by LK-Mini."""

    if not 0 <= channel_index < 16:
        raise ValueError("channel_index must be between 0 and 15")
    if gain not in GAIN_CODES:
        raise ValueError(f"Unsupported gain {gain}; choose from {tuple(GAIN_CODES)}")
    code = CHANNEL_CODES[channel_index]
    parameters = (
        f"{int(power_down)}{GAIN_CODES[gain]}0"
        f"{int(bias)}{int(srb2)}{int(srb1)}"
    )
    return f"x{code}{parameters}X"


def build_all_channel_commands(active_channels: Sequence[int], gain: int = 24) -> str:
    active = set(active_channels)
    if any(channel < 0 or channel >= 16 for channel in active):
        raise ValueError("active channel indices must be between 0 and 15")
    return "".join(
        build_channel_command(channel, gain, power_down=channel not in active)
        for channel in range(16)
    )


def build_impedance_command(channel_index: int, p_input: bool = True, n_input: bool = False) -> str:
    if not 0 <= channel_index < 16:
        raise ValueError("channel_index must be between 0 and 15")
    return f"z{CHANNEL_CODES[channel_index]}{int(p_input)}{int(n_input)}Z"


def _host_timestamps(sample_count: int, sample_rate: int) -> np.ndarray:
    if sample_count <= 0:
        return np.empty(0, dtype=np.float64)
    end = time.monotonic()
    return end - np.arange(sample_count - 1, -1, -1, dtype=np.float64) / sample_rate


class BrainFlowSource(EEGSource):
    """Direct BrainFlow connection to LK-Mini as a Cyton+Daisy Wi-Fi board."""

    def __init__(
        self,
        *,
        ip_address: str = "192.168.4.1",
        ip_port: int = 12345,
        sample_rate: int = 250,
        channel_indices: Sequence[int] = tuple(range(8)),
        channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES,
        gain: int = 24,
        timeout: int = 5,
    ):
        if sample_rate not in RATE_COMMANDS:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")
        if len(channel_indices) != len(channel_names):
            raise ValueError("channel_indices and channel_names must have equal length")
        self.ip_address = ip_address
        self.ip_port = int(ip_port)
        self.sample_rate = int(sample_rate)
        self.channel_indices = tuple(int(value) for value in channel_indices)
        self.channel_names = tuple(channel_names)
        self.gain = int(gain)
        self.timeout = int(timeout)
        self._board = None
        self._brainflow_eeg_channels: list[int] = []

    def start(self) -> None:
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ImportError as exception:
            raise RuntimeError("BrainFlow is not installed; run pip install brainflow") from exception

        params = BrainFlowInputParams()
        params.ip_address = self.ip_address
        params.ip_port = self.ip_port
        params.timeout = self.timeout
        board_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
        board = BoardShim(board_id, params)
        try:
            board.prepare_session()
            board.config_board(RATE_COMMANDS[self.sample_rate])
            time.sleep(0.25)
            board.config_board(build_all_channel_commands(self.channel_indices, self.gain))
            time.sleep(0.25)
            eeg_channels = list(BoardShim.get_eeg_channels(board_id))
            if max(self.channel_indices) >= len(eeg_channels):
                raise RuntimeError(
                    f"BrainFlow reports {len(eeg_channels)} EEG channels, "
                    f"but channel {max(self.channel_indices) + 1} was requested"
                )
            board.start_stream(45000)
            board.get_board_data()
        except BaseException:
            if board.is_prepared():
                board.release_session()
            raise
        self._board = board
        self._brainflow_eeg_channels = eeg_channels

    def read(self, timeout: float = 0.25) -> EEGChunk:
        if self._board is None:
            raise RuntimeError("Source has not been started")
        deadline = time.monotonic() + timeout
        while True:
            board_data = self._board.get_board_data()
            if board_data.shape[1]:
                rows = [self._brainflow_eeg_channels[index] for index in self.channel_indices]
                data = np.asarray(board_data[rows, :], dtype=np.float64)
                return EEGChunk(data, _host_timestamps(data.shape[1], self.sample_rate))
            if time.monotonic() >= deadline:
                return EEGChunk(
                    np.empty((len(self.channel_names), 0)), np.empty(0, dtype=np.float64)
                )
            time.sleep(0.005)

    def stop(self) -> None:
        board, self._board = self._board, None
        if board is None:
            return
        try:
            board.stop_stream()
        except BaseException:
            pass
        finally:
            if board.is_prepared():
                board.release_session()


class LSLSource(EEGSource):
    """Subscribe to an OpenBCI GUI LSL time-series stream."""

    def __init__(
        self,
        *,
        stream_type: str = "EEG",
        stream_name: str | None = None,
        sample_rate: int = 250,
        channel_indices: Sequence[int] = tuple(range(8)),
        channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES,
        resolve_timeout: float = 10.0,
    ):
        if len(channel_indices) != len(channel_names):
            raise ValueError("channel_indices and channel_names must have equal length")
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.sample_rate = int(sample_rate)
        self.channel_indices = tuple(int(value) for value in channel_indices)
        self.channel_names = tuple(channel_names)
        self.resolve_timeout = float(resolve_timeout)
        self._inlet = None

    def start(self) -> None:
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError as exception:
            raise RuntimeError("pylsl is not installed; run pip install pylsl") from exception

        property_name = "name" if self.stream_name else "type"
        property_value = self.stream_name or self.stream_type
        streams = resolve_byprop(property_name, property_value, timeout=self.resolve_timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream found with {property_name}={property_value!r}")
        inlet = StreamInlet(streams[0], max_buflen=30, recover=True)
        info = inlet.info(timeout=2.0)
        nominal_rate = int(round(info.nominal_srate())) if info.nominal_srate() else self.sample_rate
        if nominal_rate != self.sample_rate:
            raise RuntimeError(
                f"LSL stream is {nominal_rate} Hz but --sample-rate is {self.sample_rate} Hz"
            )
        if max(self.channel_indices) >= info.channel_count():
            raise RuntimeError(
                f"LSL stream has {info.channel_count()} channels, "
                f"but channel {max(self.channel_indices) + 1} was requested"
            )
        self._inlet = inlet

    def read(self, timeout: float = 0.25) -> EEGChunk:
        if self._inlet is None:
            raise RuntimeError("Source has not been started")
        samples, _lsl_timestamps = self._inlet.pull_chunk(timeout=timeout)
        if not samples:
            return EEGChunk(
                np.empty((len(self.channel_names), 0)), np.empty(0, dtype=np.float64)
            )
        all_channels = np.asarray(samples, dtype=np.float64).T
        data = all_channels[list(self.channel_indices), :]
        return EEGChunk(data, _host_timestamps(data.shape[1], self.sample_rate))

    def stop(self) -> None:
        self._inlet = None


class ReplaySource(EEGSource):
    """Replay a recorded NPZ through the same online code path."""

    def __init__(self, path: str | Path, *, realtime: bool = True, chunk_samples: int = 25):
        self.path = Path(path)
        self.realtime = realtime
        self.chunk_samples = int(chunk_samples)
        self.sample_rate = 0
        self.channel_names = ()
        self._data: np.ndarray | None = None
        self._position = 0
        self._next_emit = 0.0

    def start(self) -> None:
        with np.load(self.path, allow_pickle=False) as loaded:
            key = "continuous" if "continuous" in loaded.files else "data"
            data = np.asarray(loaded[key], dtype=np.float64)
            sample_rate = int(np.asarray(loaded["sample_rate"]).item())
            channel_names = tuple(str(value) for value in loaded["channel_names"].tolist())
        if data.ndim != 2 or data.shape[0] != len(channel_names):
            raise ValueError("Replay data must be channels x samples with matching names")
        self._data = data
        self.sample_rate = sample_rate
        self.channel_names = channel_names
        self._position = 0
        self._next_emit = time.monotonic()

    def read(self, timeout: float = 0.25) -> EEGChunk:
        if self._data is None:
            raise RuntimeError("Source has not been started")
        if self._position >= self._data.shape[1]:
            time.sleep(min(timeout, 0.02))
            return EEGChunk(np.empty((len(self.channel_names), 0)), np.empty(0))
        if self.realtime:
            delay = self._next_emit - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, timeout))
                if time.monotonic() < self._next_emit:
                    return EEGChunk(np.empty((len(self.channel_names), 0)), np.empty(0))
        stop = min(self._position + self.chunk_samples, self._data.shape[1])
        data = self._data[:, self._position:stop]
        self._position = stop
        self._next_emit = time.monotonic() + data.shape[1] / self.sample_rate
        return EEGChunk(data, _host_timestamps(data.shape[1], self.sample_rate))

    def stop(self) -> None:
        self._data = None


class SyntheticSource(EEGSource):
    """Deterministic EEG-like source for UI development without a participant."""

    def __init__(
        self,
        *,
        sample_rate: int = 250,
        channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES,
        chunk_samples: int = 25,
        realtime: bool = True,
        seed: int = 42,
    ):
        self.sample_rate = int(sample_rate)
        self.channel_names = tuple(channel_names)
        self.chunk_samples = int(chunk_samples)
        self.realtime = realtime
        self._rng = np.random.default_rng(seed)
        self._sample_index = 0
        self._next_emit = 0.0
        self._running = False

    def start(self) -> None:
        self._sample_index = 0
        self._next_emit = time.monotonic()
        self._running = True

    def read(self, timeout: float = 0.25) -> EEGChunk:
        if not self._running:
            raise RuntimeError("Source has not been started")
        if self.realtime:
            delay = self._next_emit - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, timeout))
                if time.monotonic() < self._next_emit:
                    return EEGChunk(np.empty((len(self.channel_names), 0)), np.empty(0))
        indices = self._sample_index + np.arange(self.chunk_samples)
        seconds = indices / self.sample_rate
        data = []
        for channel in range(len(self.channel_names)):
            mu = (8.0 + channel * 0.35) * np.sin(2 * np.pi * (10.0 + 0.2 * channel) * seconds)
            beta = 2.5 * np.sin(2 * np.pi * 20.0 * seconds + channel * 0.3)
            drift = 5.0 * np.sin(2 * np.pi * 0.2 * seconds)
            noise = self._rng.normal(0.0, 3.0, self.chunk_samples)
            data.append(mu + beta + drift + noise)
        self._sample_index += self.chunk_samples
        self._next_emit = time.monotonic() + self.chunk_samples / self.sample_rate
        array = np.asarray(data, dtype=np.float64)
        return EEGChunk(array, _host_timestamps(array.shape[1], self.sample_rate))

    def stop(self) -> None:
        self._running = False


def create_source(
    backend: str,
    *,
    sample_rate: int = 250,
    channel_indices: Sequence[int] = tuple(range(8)),
    channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES,
    ip_address: str = "192.168.4.1",
    ip_port: int = 12345,
    gain: int = 24,
    lsl_name: str | None = None,
    replay_path: str | Path | None = None,
    realtime: bool = True,
) -> EEGSource:
    normalized = backend.lower()
    if normalized == "brainflow":
        return BrainFlowSource(
            ip_address=ip_address,
            ip_port=ip_port,
            sample_rate=sample_rate,
            channel_indices=channel_indices,
            channel_names=channel_names,
            gain=gain,
        )
    if normalized == "lsl":
        return LSLSource(
            stream_name=lsl_name,
            sample_rate=sample_rate,
            channel_indices=channel_indices,
            channel_names=channel_names,
        )
    if normalized == "replay":
        if replay_path is None:
            raise ValueError("replay backend requires replay_path")
        return ReplaySource(replay_path, realtime=realtime)
    if normalized == "synthetic":
        return SyntheticSource(
            sample_rate=sample_rate,
            channel_names=channel_names,
            realtime=realtime,
        )
    raise ValueError(f"Unknown EEG backend: {backend}")

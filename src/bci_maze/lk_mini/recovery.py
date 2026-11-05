"""Bounded, cancellable recovery around an isolated BrainFlow session.

Only the owned worker process loads the hardware library. Native calls can block;
the reader continues to return empty, explicitly segmented chunks during recovery.
"""

from __future__ import annotations

import errno
import multiprocessing
import threading
import time
from collections import deque

import numpy as np

from bci_maze.stopping import defer_stop

from .source import BrainFlowSource, EEGChunk, EEGSource, EEGTransportError, _SourceAlreadyStarted, _brainflow_config


def is_transport_error(error: Exception) -> bool:
    """Retry recognized transport failures; programming/configuration errors fail."""
    if isinstance(error, (EEGTransportError, TimeoutError, ConnectionError)):
        return True
    if isinstance(error, OSError):
        return error.errno in {
            errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
            errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPIPE,
            10060, 10061, 10054, 10053, 10051, 10065,
        }
    try:
        from brainflow.exit_codes import BrainFlowError
    except ImportError:
        return False
    return isinstance(error, BrainFlowError) and error.exit_code in {
        1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 18,
    }


def _brainflow_worker(connection, session_parameters: dict) -> None:
    source = None
    try:
        source = BrainFlowSource(**session_parameters)
        source.start()
        connection.send(("ready",))
        while True:
            if connection.poll() and connection.recv() == "stop":
                break
            chunk = source.read(timeout=0.05)
            if chunk.data.shape[1]:
                connection.send((
                    "data", chunk.data, chunk.monotonic_timestamps,
                    chunk.received_timestamps, time.monotonic(),
                ))
    except Exception as error:
        try:
            connection.send(("error", is_transport_error(error), f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if source is not None:
            try:
                source.stop()
            except Exception:
                pass
        connection.close()


class RecoveringBrainFlowSource(EEGSource):
    """Read one independently recoverable BrainFlow source.

    ``start`` launches asynchronously. ``max_retries=0`` retries until stopped;
    positive values bound consecutive retries. Every loss of an established stream
    immediately advances ``segment_id``. Consumers must discard partial windows
    at that boundary, including when it is first reported in an empty chunk.
    """

    validates_timing = True

    def __init__(
        self, *, session_parameters: dict | None = None,
        connect_timeout: float = 12.0, no_data_timeout: float = 3.0,
        max_retries: int = 0, retry_delay: float = 0.5,
        max_retry_delay: float = 3.0, shutdown_timeout: float = 4.0,
        _context=None, _worker_target=None,
    ):
        self._session_parameters = _brainflow_config(**(session_parameters or {}))
        self.__dict__.update(self._session_parameters)
        for name, value in (
            ("connect_timeout", connect_timeout), ("no_data_timeout", no_data_timeout),
            ("shutdown_timeout", shutdown_timeout),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not np.isfinite(retry_delay) or retry_delay < 0:
            raise ValueError("retry_delay must be finite and non-negative")
        if not np.isfinite(max_retry_delay) or max_retry_delay < retry_delay:
            raise ValueError("max_retry_delay must be finite and at least retry_delay")
        self.connect_timeout = float(connect_timeout)
        self.no_data_timeout = float(no_data_timeout)
        self.max_retries = max_retries
        self.retry_delay = float(retry_delay)
        self.max_retry_delay = float(max_retry_delay)
        self.shutdown_timeout = float(shutdown_timeout)
        self._context = _context or multiprocessing.get_context("spawn")
        self._worker_target = _worker_target or _brainflow_worker
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._process = self._connection = None
        self._phase = "stopped"
        self._state = "stopped"
        self.segment_id = 0
        self._events = deque(maxlen=1000)
        self._stats = dict(attempts=0, reconnects=0, gaps=0, failures=0, forced_reconnects=0)
        self._episode_attempts = 0
        self._session_ready = False
        self._session_has_data = False
        self._has_data = False
        self._pending_recovery = False
        self._error: RuntimeError | None = None
        self._deadline = self._last_data_at = self._retry_at = 0.0
        self._kill_sent = False

    @property
    def connection_state(self) -> str:
        return self._state

    @property
    def recovering(self) -> bool:
        return self._state in ("connecting", "reconnecting")

    @property
    def recovery_stats(self) -> dict:
        return dict(self._stats)

    @property
    def recovery_events(self) -> list[dict]:
        return [dict(event) for event in self._events]

    def _event(self, kind: str, reason: str = "") -> None:
        self._events.append(dict(
            type=kind, state=self._state, segment_id=self.segment_id,
            time_monotonic=time.monotonic(), reason=reason,
            attempt=self._stats["attempts"],
        ))

    def _empty(self) -> EEGChunk:
        return EEGChunk(np.empty((len(self.channel_names), 0)), np.empty(0), segment_id=self.segment_id)

    def start(self) -> None:
        with self._lock:
            if self._phase != "stopped":
                raise _SourceAlreadyStarted("Source is already started; stop it before starting again")
            self._cancelled.clear()
            self._error = None
            self._episode_attempts = 0
            if self._stats["attempts"]:
                self.segment_id += 1
                self._pending_recovery = self._has_data
            self._state = "connecting"
            self._spawn_worker()

    @defer_stop()
    def _spawn_worker(self) -> None:
        if self._process is not None:
            raise RuntimeError("Previous EEG worker has not exited")
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._worker_target, args=(child, self._session_parameters),
            name="lk-mini-brainflow", daemon=True,
        )
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            self._phase = self._state = "failed"
            raise
        self._process, self._connection = process, parent
        child.close()
        self._stats["attempts"] += 1
        self._episode_attempts += 1
        self._phase = "starting"
        self._session_ready = False
        self._session_has_data = False
        self._deadline = time.monotonic() + self.connect_timeout
        self._event("connecting")

    def _retire_worker(self) -> None:
        self._phase = "cleanup"
        self._deadline = time.monotonic() + self.shutdown_timeout
        self._kill_sent = False
        if self._connection is not None:
            try:
                self._connection.send("stop")
            except (BrokenPipeError, EOFError, OSError):
                pass

    def _fail_session(self, reason: str, *, retryable: bool = True, forced: bool = False) -> None:
        if self._phase in ("cleanup", "backoff", "failed", "stopped"):
            return
        self._stats["forced_reconnects" if forced else "failures"] += 1
        if self._session_ready:
            self.segment_id += 1
            if self._session_has_data and not forced:
                self._stats["gaps"] += 1
            self._pending_recovery = self._has_data
        if self._episode_attempts == 0:
            self._episode_attempts = 1
        self._state = "reconnecting"
        self._event("forced_reconnect" if forced else "disconnected", reason)
        retry_exhausted = self.max_retries > 0 and self._episode_attempts > self.max_retries
        if not retryable or (retry_exhausted and not forced):
            self._error = RuntimeError(reason if not retryable else f"EEG recovery retries exhausted: {reason}")
        delay = min(self.max_retry_delay, self.retry_delay * 2 ** min(max(self._episode_attempts - 1, 0), 20))
        self._retry_at = time.monotonic() + delay
        self._retire_worker()

    def request_reconnect(self, reason: str = "requested session restart") -> None:
        """Rebuild this source's session; record separately from transport failures."""
        with self._lock:
            if self._phase in ("starting", "streaming"):
                self._fail_session(reason, forced=True)

    def _cleanup_step(self) -> None:
        process, connection = self._process, self._connection
        if process is None:
            return
        # Drain pending sends so a cooperative worker can observe the stop command.
        if connection is not None:
            try:
                for _ in range(64):
                    if not connection.poll():
                        break
                    connection.recv()
            except (EOFError, OSError):
                pass
        if process.is_alive() and time.monotonic() >= self._deadline:
            if not self._kill_sent:
                process.terminate()
                self._kill_sent = True
                self._deadline = time.monotonic() + 1.0
                self._event("worker_terminated", "native session did not stop within its deadline")
            else:
                process.kill()
                self._deadline = time.monotonic() + 1.0
        if process.is_alive():
            return
        process.join(timeout=0)
        process.close()
        connection.close()
        self._process = self._connection = None
        if self._cancelled.is_set():
            self._phase = self._state = "stopped"
        elif self._error is not None:
            self._phase = self._state = "failed"
            self._event("failed", str(self._error))
        else:
            self._phase = "backoff"

    def _step(self) -> EEGChunk | None:
        now = time.monotonic()
        if self._phase == "cleanup":
            self._cleanup_step()
        elif self._phase == "backoff" and now >= self._retry_at:
            self._spawn_worker()
        elif self._phase in ("starting", "streaming"):
            try:
                message = self._connection.recv() if self._connection.poll() else None
            except (EOFError, OSError) as error:
                self._fail_session(f"EEG worker connection closed: {error}")
                return self._empty()
            if message is not None:
                kind = message[0]
                if kind == "error":
                    self._fail_session(message[2], retryable=message[1])
                    return self._empty()
                if kind == "ready":
                    self._phase = "streaming"
                    self._session_ready = True
                    self._last_data_at = now
                    self._event("connected")
                elif kind == "data":
                    if now - message[4] > self.no_data_timeout:
                        self._fail_session("EEG worker data exceeded the freshness deadline")
                        return self._empty()
                    chunk = EEGChunk(message[1], message[2], message[3], self.segment_id)
                    self._last_data_at = now
                    self._episode_attempts = 0
                    self._has_data = True
                    self._session_has_data = True
                    self._state = "streaming"
                    if self._pending_recovery:
                        self._stats["reconnects"] += 1
                        self._pending_recovery = False
                        self._event("recovered")
                    return chunk
                else:
                    self._fail_session(f"Unknown EEG worker message: {kind}", retryable=False)
                    return self._empty()
            if self._phase == "starting" and now >= self._deadline:
                self._fail_session("BrainFlow connection deadline exceeded")
                return self._empty()
            if self._phase == "streaming" and now - self._last_data_at >= self.no_data_timeout:
                self._fail_session(f"No EEG samples for {self.no_data_timeout:g} seconds")
                return self._empty()
            if not self._process.is_alive() and not self._connection.poll():
                self._fail_session(f"EEG worker exited with code {self._process.exitcode}")
                return self._empty()
        if self._phase == "failed":
            raise self._error or RuntimeError("EEG worker failed")
        return None

    def read(self, timeout: float = 0.25) -> EEGChunk:
        if not np.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._cancelled.is_set():
                    return self._empty()
                if self._phase == "stopped":
                    raise RuntimeError("Source has not been started")
                chunk = self._step()
                if chunk is not None:
                    return chunk
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._cancelled.wait(min(0.005, remaining)):
                return self._empty()

    @defer_stop()
    def stop(self) -> None:
        self._cancelled.set()
        with self._lock:
            if self._process is None:
                self._phase = self._state = "stopped"
                return
            self._retire_worker()
        # Native stop/release has its own deadline; only this owned worker is killed.
        deadline = time.monotonic() + self.shutdown_timeout + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                self._cleanup_step()
                if self._process is None:
                    self._event("stopped")
                    return
            time.sleep(0.01)
        raise RuntimeError("Owned EEG worker did not exit after termination")

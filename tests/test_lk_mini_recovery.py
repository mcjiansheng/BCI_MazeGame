from collections import deque
import argparse
import errno
import signal
import time
import types

import numpy as np
import pytest

from bci_maze.lk_mini import recovery
from bci_maze.lk_mini.cli import add_source_arguments, source_from_args
from bci_maze.lk_mini.recovery import RecoveringBrainFlowSource, is_transport_error
from bci_maze.lk_mini.source import BrainFlowSource, EEGTransportError


class FakeContext:
    """In-process protocol peer; all waits are controlled explicitly by the test."""

    def __init__(self, sessions, *, hung=False, interrupt_start=False):
        self.sessions = deque(sessions)
        self.hung = hung
        self.interrupt_start = interrupt_start
        self.processes = []
        self.parent = None

    def Pipe(self, duplex=True):
        context = self

        class Connection:
            def __init__(self):
                self.messages = deque()
                self.closed = False
                self.process = None

            def poll(self):
                return bool(self.messages)

            def recv(self):
                return self.messages.popleft()

            def send(self, message):
                assert message == "stop"
                if not context.hung:
                    self.process.alive = False

            def close(self):
                self.closed = True

        self.parent = Connection()
        return self.parent, Connection()

    def Process(self, **kwargs):
        context = self

        class Process:
            exitcode = 0

            def __init__(self):
                self.alive = False
                self.closed = False
                self.terminated = False
                self.connection = context.parent
                self.connection.process = self

            def start(self):
                assert all(not existing.alive for existing in context.processes)
                self.alive = True
                self.connection.messages.extend(context.sessions.popleft())
                context.processes.append(self)
                if context.interrupt_start:
                    signal.raise_signal(signal.SIGINT)

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def kill(self):
                self.terminate()

            def join(self, timeout=0):
                pass

            def close(self):
                self.closed = True

        return Process()


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr(recovery, "time", types.SimpleNamespace(
        monotonic=lambda: value[0], sleep=lambda duration: value.__setitem__(0, value[0] + duration),
    ))
    return value


def ready():
    return ("ready",)


def data(at=100.0):
    return ("data", np.ones((16, 2)), np.asarray([at, at + 0.004]), np.asarray([1.0, 1.004]), at)


def source_for(sessions, **kwargs):
    context = FakeContext(sessions)
    source = RecoveringBrainFlowSource(
        _context=context, retry_delay=0, max_retry_delay=0, **kwargs,
    )
    return source, context


def next_data(source, limit=20):
    for _ in range(limit):
        chunk = source.read(timeout=0)
        if chunk.data.shape[1]:
            return chunk
    raise AssertionError("No data after expected recovery")


def test_cli_enables_async_recovery_and_can_select_single_session():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    source = source_from_args(parser.parse_args([]))
    assert isinstance(source, RecoveringBrainFlowSource)
    assert source.max_retries == 0
    assert isinstance(source_from_args(parser.parse_args(["--no-auto-reconnect"])), BrainFlowSource)


@pytest.mark.parametrize("error,retry", [
    (EEGTransportError("package discontinuity"), True),
    (TimeoutError("timed out"), True),
    (ConnectionResetError("reset"), True),
    (OSError(errno.ECONNREFUSED, "refused"), True),
    (OSError(errno.ENOENT, "missing file"), False),
    (ValueError("bad channels"), False),
    (RuntimeError("programming error"), False),
])
def test_only_transport_errors_are_retried(error, retry):
    assert is_transport_error(error) is retry


def test_native_configuration_and_session_conflicts_fail_fast():
    from brainflow.exit_codes import BrainFlowError
    assert is_transport_error(BrainFlowError("write failed", 4))
    assert is_transport_error(BrainFlowError("timeout", 18))
    assert not is_transport_error(BrainFlowError("invalid argument", 13))
    assert not is_transport_error(BrainFlowError("another board", 16))


def test_initial_failure_retries_without_claiming_recovery_or_gap(clock):
    source, context = source_for([
        [("error", True, "connect timeout")], [ready(), data()],
    ])
    with source:
        assert source.recovering and source.connection_state == "connecting"
        first = next_data(source)
        assert first.segment_id == 0
        assert not source.recovering
        assert source.recovery_stats == dict(attempts=2, reconnects=0, gaps=0, failures=1, forced_reconnects=0)
    assert all(process.closed for process in context.processes)
    assert source.recovery_stats["attempts"] == 2


def test_disconnect_invalidates_segment_before_new_samples_and_preserves_gap(clock):
    source, context = source_for([
        [ready(), data(), ("error", True, "package discontinuity")],
        [ready(), data(at=104.0)],
    ])
    with source:
        first = next_data(source)
        clock[0] = 104.0
        empty = source.read(timeout=0)
        assert empty.data.shape[1] == 0
        assert empty.segment_id == source.segment_id == 1
        assert source.recovering
        assert source.recovery_stats["reconnects"] == 0
        second = next_data(source)
        assert second.segment_id == 1
        assert second.monotonic_timestamps[0] - first.monotonic_timestamps[-1] > 3.9
        assert source.recovery_stats["reconnects"] == 1
        assert source.recovery_stats["gaps"] == 1
        assert len(context.processes) == 2


def test_no_data_timeout_reconnects_and_does_not_count_ready_as_recovered(clock):
    source, _ = source_for([[ready(), data()], [ready(), data(at=104.0)]], no_data_timeout=3)
    with source:
        next_data(source)
        clock[0] = 104.0
        assert source.read(timeout=0).segment_id == 1
        source.read(timeout=0)  # release old worker
        source.read(timeout=0)  # start new worker
        source.read(timeout=0)  # receive ready
        assert source.recovery_stats["reconnects"] == 0
        assert source.recovering
        assert next_data(source).segment_id == 1
        assert source.recovery_stats["reconnects"] == 1


def test_config_error_is_not_retried_and_original_reason_survives_cleanup(clock):
    source, context = source_for([[("error", False, "invalid channel configuration")]])
    with source:
        source.read(timeout=0)
        with pytest.raises(RuntimeError, match="invalid channel configuration"):
            source.read(timeout=0)
        assert len(context.processes) == 1


def test_finite_retry_budget_applies_to_initialization(clock):
    source, context = source_for([
        [("error", True, "first")], [("error", True, "second")],
    ], max_retries=1)
    with source:
        with pytest.raises(RuntimeError, match="retries exhausted: second"):
            for _ in range(10):
                source.read(timeout=0)
        assert len(context.processes) == 2


def test_configuration_is_validated_before_starting_worker():
    with pytest.raises(ValueError, match="sample rate"):
        RecoveringBrainFlowSource(session_parameters={"sample_rate": 123})
    with pytest.raises(ValueError, match="no_data_timeout"):
        RecoveringBrainFlowSource(no_data_timeout=float("nan"))


@pytest.mark.parametrize(("parameters", "field"), [
    ({"gain": 24.5}, "gain"),
    ({"gain": True}, "gain"),
    ({"ip_port": 12345.25}, "ip_port"),
    ({"ip_port": False}, "ip_port"),
    ({"sample_rate": 250.5}, "sample rate"),
])
def test_fractional_and_boolean_board_settings_fail_before_start(parameters, field):
    with pytest.raises(ValueError, match=field):
        RecoveringBrainFlowSource(session_parameters=parameters)


def test_integer_valued_float_settings_are_preserved():
    source = RecoveringBrainFlowSource(session_parameters={
        "gain": 24.0, "ip_port": 12345.0, "sample_rate": 250.0,
    })
    assert (source.gain, source.ip_port, source.sample_rate) == (24, 12345, 250)


def test_interrupt_after_process_start_closes_owned_worker(clock):
    context = FakeContext([[ready()]], interrupt_start=True)
    source = RecoveringBrainFlowSource(_context=context)
    with pytest.raises(KeyboardInterrupt):
        with source:
            pytest.fail("cancelled startup entered the source context")
    assert len(context.processes) == 1
    assert context.processes[0].closed and not context.processes[0].alive
    assert context.parent.closed
    assert source.connection_state == "stopped"


def test_rejected_nested_start_leaves_outer_session_running(clock):
    source, _ = source_for([[ready(), data()]])
    with source:
        with pytest.raises(RuntimeError, match="already started"):
            with source:
                pytest.fail("a nested context acquired the same source")
        assert next_data(source).data.shape[1] == 2


def test_stop_during_backoff_prevents_new_session(clock):
    source, context = source_for([[("error", True, "offline")]])
    source.start()
    source.read(timeout=0)
    source.read(timeout=0)
    source.stop()
    clock[0] += 100
    assert source.read(timeout=0).data.shape[1] == 0
    assert len(context.processes) == 1


def test_forced_reconnect_is_identified_separately(clock):
    source, _ = source_for([[ready(), data()], [ready(), data(at=101.0)]])
    with source:
        next_data(source)
        source.request_reconnect("test injection")
        assert source.segment_id == 1 and source.recovering
        assert source.recovery_stats["failures"] == 0
        assert source.recovery_stats["gaps"] == 0
        assert source.recovery_stats["forced_reconnects"] == 1
        clock[0] = 101
        assert next_data(source).segment_id == 1
        assert any(event["type"] == "forced_reconnect" for event in source.recovery_events)


def test_initialization_deadline_does_not_start_overlapping_workers(clock):
    context = FakeContext([[], [ready(), data(at=103)]], hung=True)
    source = RecoveringBrainFlowSource(
        _context=context, connect_timeout=1, shutdown_timeout=1,
        retry_delay=0, max_retry_delay=0,
    )
    source.start()
    clock[0] = 101
    assert source.read(timeout=0).data.shape[1] == 0
    assert len(context.processes) == 1 and context.processes[0].alive
    clock[0] = 102
    source.read(timeout=0)
    assert context.processes[0].terminated
    assert len(context.processes) == 1
    clock[0] = 103
    next_data(source)
    assert len(context.processes) == 2
    source.stop()
    assert all(not process.alive for process in context.processes)


def _hung_worker(connection, parameters):
    while True:
        time.sleep(60)


def test_actual_owned_process_can_be_cancelled_while_initializing():
    source = RecoveringBrainFlowSource(
        _worker_target=_hung_worker, connect_timeout=30, shutdown_timeout=0.05,
    )
    started = time.monotonic()
    source.start()
    assert time.monotonic() - started < 2
    try:
        began_read = time.monotonic()
        assert source.read(timeout=0.02).data.shape[1] == 0
        assert time.monotonic() - began_read < 0.5
    finally:
        began_stop = time.monotonic()
        source.stop()
    assert time.monotonic() - began_stop < 2
    assert source._process is None
    assert source.connection_state == "stopped"
    assert any(event["type"] == "worker_terminated" for event in source.recovery_events)

import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from bci_maze.lk_mini.game_control import (
    GameControlWorker, JunctionController, MazeUdpClient, model_control_provenance,
)


class Maze:
    def __init__(self):
        self.nav = dict(mode="first_person_relative", maze_generation=1, state="Junction", paused=False,
                        x=3, y=5, heading=0, steps=8, can_left=True, can_right=True,
                        can_forward=False, can_uturn=True, pending_turn="")
        self.commands = []
        self.fail = ""

    def request(self, command, *, expected=None):
        self.commands.append(command)
        if expected is not None and any(self.nav.get(key) != value for key, value in expected.items()):
            raise ValueError("Unity rejected stale conditional command")
        if command == self.fail:
            raise TimeoutError("missing ACK")
        if command == "pause":
            self.nav["paused"] = True
        elif command == "resume":
            self.nav["paused"] = False
        elif command == "cancel":
            self.nav["pending_turn"] = ""
        elif command in {"left", "right", "uturn"}:
            self.nav["pending_turn"] = command
        elif command == "forward":
            self.nav.update(paused=False, state="Moving", pending_turn="")
        return dict(self.nav)


def test_junction_pauses_accepts_one_choice_and_recognizes_revisit():
    maze = Maze()
    control = JunctionController(maze.request)
    snapshot = control.step(True)
    assert snapshot.phase == "waiting"
    assert maze.nav["paused"]
    token = snapshot.visit_token
    assert control.step(True, (token, "left")).phase == "submitted"
    assert maze.commands[-2:] == ["left", "resume"]
    control.step(True, (token, "right"))
    assert maze.commands.count("left") == 1 and "right" not in maze.commands
    assert maze.commands.count("pause") == 1
    maze.nav.update(state="Moving", steps=9)
    control.step(True)
    maze.nav.update(state="Junction", steps=30)
    revisit = control.step(True)
    assert revisit.visit_token != token
    assert revisit.window_id > snapshot.window_id


def test_dynamic_two_choice_mapping_supports_forward_without_extra_resume():
    maze = Maze()
    maze.nav.update(can_left=False, can_forward=True)
    control = JunctionController(maze.request)
    waiting = control.step(True)
    assert waiting.command_for("left_hand") == "forward"
    assert waiting.command_for("right_hand") == "right"
    control.step(True, (waiting.visit_token, "forward"))
    assert maze.commands[-1] == "forward"
    assert "resume" not in maze.commands
    assert not maze.nav["paused"]


def test_reacquiring_pause_at_same_cell_requires_new_window_and_intent():
    maze = Maze()
    control = JunctionController(maze.request)
    first = control.step(True)
    maze.nav["paused"] = False  # The player resumed while the classifier was warming up.
    second = control.step(True, (first.visit_token, "left"))
    assert second.phase == "waiting"
    assert second.window_id > first.window_id
    assert second.visit_token != first.visit_token
    assert "left" not in maze.commands and "resume" not in maze.commands
    assert maze.commands.count("pause") == 2


@pytest.mark.parametrize("changed", [
    {"x": 7, "y": 9, "heading": 2, "steps": 0},
    {"paused": False},
    {"state": "Moving"},
])
def test_turn_ack_must_confirm_original_paused_visit_or_cancel(changed):
    maze = Maze()
    original = maze.request
    def race(command, **kwargs):
        nav = original(command, **kwargs)
        if command == "left":
            maze.nav.update(changed)
            nav = dict(maze.nav)
        return nav
    control = JunctionController(race)
    waiting = control.step(True)
    result = control.step(True, (waiting.visit_token, "left"))
    assert result.phase == "unavailable"
    assert "active junction" in result.error
    assert maze.commands[-1] == "cancel"
    assert maze.nav["pending_turn"] == ""
    assert "resume" not in maze.commands


def test_junction_requires_generation_but_continuous_keeps_legacy_protocol():
    maze = Maze()
    del maze.nav["maze_generation"]
    result = JunctionController(maze.request).step(True)
    assert result.phase == "unavailable"
    assert "updated Unity" in result.error
    assert maze.commands == ["status"]
    assert JunctionController(maze.request, mode="continuous").step(True).phase == "continuous"


def test_atomic_condition_rejects_restart_before_turn_without_applying_intent():
    maze = Maze()
    original = maze.request
    applied = []
    def race(command, **kwargs):
        if command == "left":
            maze.nav["maze_generation"] += 1
        nav = original(command, **kwargs)
        if command == "left":
            applied.append(command)
        return nav
    controller = JunctionController(race)
    waiting = controller.step(True)
    assert controller.step(True, (waiting.visit_token, "left")).phase == "unavailable"
    assert applied == []
    assert maze.nav["pending_turn"] == ""
    assert "resume" not in maze.commands


def test_stream_recovery_cancels_intent_and_requires_new_visit_window():
    maze = Maze()
    control = JunctionController(maze.request)
    old = control.step(True)
    suspended = control.step(False, (old.visit_token, "right"))
    assert suspended.phase == "signal_wait"
    assert "right" not in maze.commands
    recovered = control.step(True, (old.visit_token, "right"))
    assert recovered.phase == "waiting"
    assert recovered.window_id > old.window_id
    assert recovered.visit_token != old.visit_token
    assert "right" not in maze.commands


def test_unknown_state_bad_ack_and_closed_exit_never_trigger_motion():
    maze = Maze()
    control = JunctionController(maze.request)
    waiting = control.step(True)
    maze.fail = "status"
    assert control.step(True, (waiting.visit_token, "left")).phase == "unavailable"
    assert "left" not in maze.commands
    maze.fail = ""
    maze.nav["state"] = "unknown"
    assert control.step(True).phase == "unavailable"
    maze.nav.update(state="Junction", can_left=False)
    waiting = control.step(True)
    assert control.step(True, (waiting.visit_token, "left")).phase == "waiting"
    assert "left" not in maze.commands
    maze.fail = "right"
    assert control.step(True, (waiting.visit_token, "right")).phase == "unavailable"
    assert "resume" not in maze.commands


def test_complete_and_cleanup_preserve_manual_pause_and_release_owned_pause():
    maze = Maze()
    maze.nav["paused"] = True
    control = JunctionController(maze.request)
    assert control.step(True).phase == "manual_pause"
    control.close()
    assert "resume" not in maze.commands
    maze.nav["paused"] = False
    control = JunctionController(maze.request)
    control.step(True)
    control.close()
    assert maze.commands[-2:] == ["cancel", "resume"]
    maze.nav["state"] = "Complete"
    assert control.step(True).phase == "complete"


@pytest.mark.parametrize("provenance", [{}, {"data_origin": "public"}, {"data_origin": "synthetic", "calibration_passed": True}])
def test_unverified_models_need_explicit_control_override(provenance):
    bundle = SimpleNamespace(provenance=provenance)
    with pytest.raises(ValueError, match="recorded EEG"):
        model_control_provenance(bundle)
    control = model_control_provenance(bundle, allow_unvalidated=True)
    assert control["unvalidated_control_override"]
    assert not control["control_validated"]
    assert control["calibration_passed"] is provenance.get("calibration_passed", False)


@pytest.mark.parametrize("origin", ["hardware_recorded", "lsl_recorded"])
@pytest.mark.parametrize("passed", [True, False])
def test_recorded_imagery_can_control_with_actual_calibration_result(origin, passed):
    bundle = SimpleNamespace(provenance=dict(data_origin=origin, calibration_passed=passed))
    control = model_control_provenance(bundle)
    assert control["calibration_passed"] is passed
    assert control["control_validated"] is passed
    assert not control["unvalidated_control_override"]


def test_motor_execution_is_not_a_game_control_task():
    bundle = SimpleNamespace(provenance=dict(data_origin="hardware_recorded", task="motor_execution"))
    with pytest.raises(ValueError, match="motor-imagery model"):
        model_control_provenance(bundle, allow_unvalidated=True)


@pytest.fixture
def udp_maze():
    """Real loopback UDP transport with a protocol simulator, without Unity or EEG."""
    maze = Maze()
    stop = threading.Event()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.05)
    maze.bad_echo = False

    def serve():
        while not stop.is_set():
            try:
                data, remote = sock.recvfrom(8192)
            except socket.timeout:
                continue
            message = json.loads(data)
            try:
                navigation = maze.request(message["command"], expected=message.get("expected"))
                accepted = True
            except ValueError:
                navigation, accepted = dict(maze.nav), False
            response = dict(command=message["command"], accepted=accepted,
                            echo_ts="wrong" if maze.bad_echo else message["ts"], navigation=navigation)
            sock.sendto(json.dumps(response).encode(), remote)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield maze, sock.getsockname()[1]
    finally:
        stop.set()
        thread.join(1)
        sock.close()


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("worker state did not arrive")


def test_udp_checks_echo_and_worker_does_not_block_reads(udp_maze):
    maze, port = udp_maze
    client = MazeUdpClient("127.0.0.1", port)
    assert client.request("status")["state"] == "Junction"
    maze.bad_echo = True
    with pytest.raises(ValueError, match="does not match"):
        client.request("status")
    maze.bad_echo = False
    worker = GameControlWorker("127.0.0.1", port, poll_seconds=0.01)
    worker.start()
    try:
        wait_for(lambda: worker.snapshot.phase == "signal_wait")
        worker.set_stream_ready(True)
        wait_for(lambda: worker.snapshot.phase == "waiting")
        token = worker.snapshot.visit_token
        assert worker.submit("left", token)
        wait_for(lambda: worker.snapshot.phase == "submitted")
        assert not worker.submit("right", token)
        assert maze.commands.count("left") == 1
        assert any(event.get("command") == "left" for event in worker.drain_events())
        maze.bad_echo = True
        wait_for(lambda: worker.snapshot.phase == "unavailable")
        start = time.monotonic()
        for _ in range(100):
            worker.snapshot
            worker.set_stream_ready(False)
        assert time.monotonic() - start < 0.1
    finally:
        maze.bad_echo = False
        worker.close()
    assert not worker._thread.is_alive()

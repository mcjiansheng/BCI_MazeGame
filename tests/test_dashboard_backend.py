"""Behavior checks for local monitoring, lifecycle, validation and HTTP isolation."""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from bci_maze.dashboard.controller import ConflictError, DashboardController, ROOT, RECOVERY_PROFILES
from bci_maze.dashboard.server import DashboardServer
from bci_maze.lk_mini.source import EEGChunk, EEGSource


def test_ui_launcher_runs_server_in_process(monkeypatch):
    from scripts import run_lk_mi
    from bci_maze.dashboard import server

    received = []
    monkeypatch.setattr(sys, "argv", ["run_lk_mi.py", "ui", "--no-browser", "--port", "8766"])
    monkeypatch.setattr(server, "main", lambda args: received.append(args) or 0)
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: pytest.fail("UI must own shutdown in the launcher process"))
    assert run_lk_mi.main() == 0
    assert received == [["--no-browser", "--port", "8766"]]


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Expected state did not arrive")


class ControlledSource(EEGSource):
    sample_rate = 250
    channel_names = ("C3", "C4")

    def __init__(self):
        self.chunks = queue.Queue()
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self, timeout=0.25):
        try:
            chunk = self.chunks.get(timeout=min(timeout, 0.03))
            self.segment_id = chunk.segment_id
            return chunk
        except queue.Empty:
            return EEGChunk(np.empty((2, 0)), np.empty(0), segment_id=self.segment_id)

    def push(self, samples=500, segment=0):
        timestamps = np.arange(samples) / self.sample_rate
        waves = np.stack([20 * np.sin(timestamps * 2 * np.pi * frequency) for frequency in (10, 12)])
        self.chunks.put(EEGChunk(waves, timestamps, segment_id=segment))


def test_monitor_quality_boundaries_and_exclusive_ownership():
    source = ControlledSource()
    controller = DashboardController(source_factory=lambda _: source)
    try:
        controller.start_monitor({"backend": "synthetic"})
        wait_for(lambda: source.started)
        assert controller.snapshot()["monitor"]["state"] == "connecting"
        source.push(1250)
        state = wait_for(lambda: controller.snapshot()["monitor"]["quality"])
        assert [item["status"] for item in state] == ["good", "good"]
        assert len(controller.snapshot()["monitor"]["waveforms"][0]) == 1250
        assert controller.snapshot()["monitor"]["waveform_seconds"] == 5
        time.sleep(0.1)  # Empty read batches retain the established stream state.
        assert controller.snapshot()["monitor"]["state"] == "streaming"
        with pytest.raises(ConflictError):
            controller.start_monitor({"backend": "synthetic"})
        source.push(100, segment=1)
        wait_for(lambda: controller.snapshot()["monitor"]["segment_id"] == 1)
        assert controller.snapshot()["monitor"]["quality"] == []
        source.push(400, segment=1)
        wait_for(lambda: bool(controller.snapshot()["monitor"]["quality"]))
        controller.stop_monitor()
        wait_for(lambda: source.stopped)
        final = controller.snapshot()["monitor"]
        assert final["state"] == "stopped"
        assert final["stop_requested"] is False
        assert final["quality"] == []
        assert final["waveforms"] == []
        controller.start_monitor({"backend": "synthetic"})
        wait_for(lambda: controller.snapshot()["monitor"]["state"] == "connecting")
    finally:
        assert controller.close()


def test_stale_data_clears_quality_and_restarts_window():
    source = ControlledSource()
    controller = DashboardController(source_factory=lambda _: source)
    try:
        controller.start_monitor({"backend": "synthetic"})
        source.push()
        wait_for(lambda: bool(controller.snapshot()["monitor"]["quality"]))
        wait_for(lambda: controller.snapshot()["monitor"]["state"] == "recovering", timeout=3)
        assert controller.snapshot()["monitor"]["quality"] == []
        source.push(100)
        wait_for(lambda: controller.snapshot()["monitor"]["samples"] == 600)
        assert controller.snapshot()["monitor"]["quality"] == []
    finally:
        controller.close()


@pytest.mark.parametrize("payload", [
    {"backend": "shell"}, {"backend": []}, {"sample_rate": 300}, {"sample_rate": True},
    {"ip_port": 0}, {"ip_port": "12345"}, {"ip_address": "127.0.0.1;cmd"},
    {"command": "rm"}, {"backend": "replay", "replay": "../../secret.npz"},
    {"recovery_profile": "turbo"}, {"recovery_profile": {}},
])
def test_reject_invalid_monitor_input(payload):
    controller = DashboardController()
    with pytest.raises(ValueError):
        controller.start_monitor(payload)
    assert controller.snapshot()["monitor"]["state"] == "idle"


def make_job_root(tmp_path):
    for folder in ("data/recordings/S01", "outputs/pretrained", "scripts", ".venv/Scripts", ".venv-mi/Scripts"):
        (tmp_path / folder).mkdir(parents=True)
    for name in ("collect_lk_mi.py", "train_lk_mi.py", "finetune_lk_semhaf.py", "online_lk_mi.py"):
        (tmp_path / "scripts" / name).touch()
    runtime = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    for name in (".venv", ".venv-mi"):
        candidate = tmp_path / name / runtime
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch()
    (tmp_path / "data/recordings/S01/cal.npz").touch()
    (tmp_path / "outputs/pretrained/base.pt").touch()
    return DashboardController(tmp_path)


def test_builds_unique_validated_commands_and_selects_runtime(tmp_path):
    controller = make_job_root(tmp_path)
    payload = dict(kind="collect", subject="S01", session="session1", backend="synthetic")
    command, first = controller._build_job(payload)
    _, second = controller._build_job(payload)
    assert first != second
    assert ".venv-mi" in command[0]
    assert "--simulate-mi" not in command
    command, output = controller._build_job(dict(kind="finetune", subject="S01", recordings=["data/recordings/S01/cal.npz"], model="outputs/pretrained/base.pt"))
    assert ".venv" in command[0] and ".venv-mi" not in command[0]
    assert "--pretrained" in command
    assert output.suffix == ".pt"
    command, _ = controller._build_job(dict(kind="online", model="outputs/pretrained/base.pt", backend="synthetic", unity_control=True))
    assert "--unity-host" in command
    assert "--allow-unvalidated-control" not in command
    with pytest.raises(ValueError):
        controller._build_job({**payload, "subject": "../escape"})
    with pytest.raises(ValueError):
        controller._input_path("scripts/collect_lk_mi.py", "recording")


@pytest.mark.parametrize("profile", ["standard", "fast"])
def test_recovery_profile_reaches_monitor_and_both_live_job_commands(tmp_path, profile):
    controller = make_job_root(tmp_path)
    source = ControlledSource()
    captured = {}

    def factory(args):
        captured.update(vars(args))
        return source

    controller._source_factory = factory
    try:
        controller.start_monitor({"backend": "brainflow", "recovery_profile": profile})
        wait_for(lambda: source.started)
        monitor = controller.snapshot()["monitor"]
        assert monitor["recovery_profile"] == profile
        for name, value in RECOVERY_PROFILES[profile].items():
            assert captured[name] == value
            assert monitor["connection_parameters"][name] == value
        assert monitor["connection_parameters"]["auto_reconnect"] is True
    finally:
        assert controller.close()
    assert controller.snapshot()["monitor"]["connection_parameters"] == monitor["connection_parameters"]
    for kind in ("collect", "online"):
        payload = dict(kind=kind, subject="S01", session="profile", backend="brainflow",
                       recovery_profile=profile, model="outputs/pretrained/base.pt")
        command, _ = controller._build_job(payload)
        for name, value in RECOVERY_PROFILES[profile].items():
            flag = "--" + name.replace("_", "-")
            assert float(command[command.index(flag) + 1]) == value


def test_source_recovery_telemetry_is_bounded_and_distinct_from_display_freshness():
    class EvidenceSource(ControlledSource):
        @property
        def connection_state(self):
            return "stopped" if self.stopped else "streaming"

        @property
        def recovery_events(self):
            return [{"type": "connecting", "index": i} for i in range(100)]

    source = EvidenceSource()
    controller = DashboardController(source_factory=lambda _: source)
    try:
        controller.start_monitor({"backend": "synthetic"})
        source.push()
        wait_for(lambda: bool(controller.snapshot()["monitor"]["quality"]))
        with controller._lock:
            controller._monitor_last = time.monotonic() - 2
        monitor = controller.snapshot()["monitor"]
        assert monitor["state"] == "recovering"
        assert monitor["source_connection_state"] == "streaming"
        assert len(monitor["recovery_events"]) == 50
        assert monitor["recovery_events"][0]["index"] == 50
    finally:
        assert controller.close()
    monitor = controller.snapshot()["monitor"]
    assert monitor["source_connection_state"] == "stopped"
    assert len(monitor["recovery_events"]) == 50


def test_busy_until_previous_log_reader_has_finished():
    controller = DashboardController()
    release = threading.Event()
    controller._job_thread = threading.Thread(target=release.wait)
    controller._job_thread.start()
    try:
        with pytest.raises(ConflictError):
            controller.start_monitor({"backend": "synthetic"})
    finally:
        release.set()
        controller._job_thread.join()


def test_participant_escape_exit_is_stopped_instead_of_failed(tmp_path):
    import io
    from types import SimpleNamespace

    controller = DashboardController(tmp_path)
    controller._job.update(state="running", kind="experiment_collect")
    process = SimpleNamespace(stdout=io.StringIO("saved partial recording\n"), wait=lambda: 130)
    controller._read_job(process, tmp_path / "unused-stop.signal")
    assert controller._job["state"] == "stopped"
    assert controller._job["error"] == ""
    assert controller._job["exit_code"] == 130


@pytest.fixture
def http_dashboard():
    controller = DashboardController()
    server = DashboardServer(("127.0.0.1", 0), controller)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, controller
    server.shutdown()
    server.server_close()
    controller.close()
    thread.join()


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = json.loads(response.read())
    status = response.status
    connection.close()
    return status, data


def test_http_same_origin_and_payload_validation(http_dashboard):
    server, _ = http_dashboard
    status, state = request(server, "GET", "/api/state")
    assert status == 200 and state["monitor"]["state"] == "idle"
    status, _ = request(server, "POST", "/api/monitor/start", '{"backend":"synthetic"}', {"Content-Type": "application/json", "Origin": "https://unrelated.invalid"})
    assert status == 403
    status, _ = request(server, "GET", "/api/state", headers={"Host": "unrelated.invalid"})
    assert status == 403
    status, _ = request(server, "POST", "/api/monitor/start", "{}", {"Content-Type": "text/plain"})
    assert status == 415
    status, _ = request(server, "POST", "/api/monitor/start", "[]", {"Content-Type": "application/json"})
    assert status == 400
    status, _ = request(server, "POST", "/api/monitor/start", '{"backend":"synthetic"}', {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{server.server_port}"})
    assert status == 200
    status, state = request(server, "POST", "/api/monitor/stop", "{}", {"Content-Type": "application/json"})
    assert status == 200 and (state["monitor"]["stop_requested"] or state["monitor"]["state"] == "stopped")


@pytest.mark.parametrize("mode", ["active", "save", "temporary_close"])
def test_worker_cooperative_stop_preserves_save_and_runs_cleanup(tmp_path, mode):
    script = tmp_path / "worker_case.py"
    ready, result, cleaned, stop = (tmp_path / name for name in ("ready", "result.npz", "cleaned", "stop"))
    if mode == "save":
        code = """
import numpy as np
import time
from pathlib import Path
def save_mi_recording():
    Path(READY).touch()
    time.sleep(1)
    np.savez_compressed(RESULT, values=np.arange(1000))
save_mi_recording()
"""
    elif mode == "temporary_close":
        code = """
import time
from pathlib import Path
def close():
    Path(READY).touch()
    time.sleep(0.8)
try:
    close()
    while True:
        time.sleep(0.05)
finally:
    time.sleep(0.5)
    Path(CLEANED).touch()
"""
    else:
        code = """
import time
from pathlib import Path
try:
    Path(READY).touch()
    while True:
        time.sleep(0.05)
finally:
    time.sleep(0.5)
    Path(CLEANED).touch()
"""
    code = code.replace("READY", repr(str(ready))).replace("RESULT", repr(str(result))).replace("CLEANED", repr(str(cleaned)))
    script.write_text(code, encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen([sys.executable, "-m", "bci_maze.dashboard.job_worker", str(stop), str(script)],
                               env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        wait_for(ready.exists)
        stop.touch()
        output, _ = process.communicate(timeout=10)
        if mode == "save":
            assert process.returncode == 0, output
            with np.load(result, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["values"], np.arange(1000))
        else:
            assert process.returncode == 130, output
            assert cleaned.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

"""Confirmed, nonblocking Unity control at one motor-imagery decision per junction."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import socket
import threading
import time
from typing import Callable
import uuid


def validated_navigation(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("mode") != "first_person_relative":
        raise ValueError("Unity did not return first-person navigation state")
    if value.get("state") not in {"Moving", "Turning", "Junction", "DeadEnd", "Complete"}:
        raise ValueError("Unknown Unity navigation state")
    for key in ("x", "y", "heading", "steps"):
        if type(value.get(key)) is not int:
            raise ValueError(f"Unity navigation is missing integer {key}")
    if not 0 <= value["heading"] <= 3 or value["steps"] < 0:
        raise ValueError("Invalid Unity navigation counters")
    for key in ("paused", "can_left", "can_right", "can_forward", "can_uturn"):
        if type(value.get(key)) is not bool:
            raise ValueError(f"Unity navigation is missing boolean {key}")
    return dict(value)


class MazeUdpClient:
    """Each request uses its own socket and a unique echoed request identifier."""

    def __init__(self, host: str, port: int, timeout: float = 0.15):
        self.address = (socket.gethostbyname(host), int(port))
        self.timeout = float(timeout)

    def request(self, command: str, *, expected: dict[str, object] | None = None) -> dict[str, object]:
        stamp = uuid.uuid4().hex
        payload = {"command": command, "ts": stamp}
        if expected is not None:
            payload["expected"] = expected
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.address)
            sock.send(json.dumps(payload).encode("utf-8"))
            reply = json.loads(sock.recv(8192).decode("utf-8"))
        if not isinstance(reply, dict) or reply.get("command") != command or reply.get("echo_ts") != stamp:
            raise ValueError("Unity ACK does not match the request")
        if reply.get("accepted") is not True:
            raise ValueError(f"Unity rejected {command}")
        return validated_navigation(reply.get("navigation"))


@dataclass(frozen=True)
class ControlSnapshot:
    phase: str = "unavailable"
    window_id: int = 0
    visit_token: tuple[int, ...] | None = None
    navigation: dict[str, object] = field(default_factory=dict)
    error: str = ""
    window_started: float = 0.0

    @property
    def accepting(self) -> bool:
        return self.phase in {"waiting", "continuous"} and self.visit_token is not None

    def command_for(self, label: str) -> str | None:
        command = {"left_hand": "left", "right_hand": "right", "feet": "uturn"}.get(label)
        if self.phase == "waiting" and command in {"left", "right"}:
            if not self.navigation.get(f"can_{command}") and self.navigation.get("can_forward"):
                return "forward"
        return command


class JunctionController:
    """Single-threaded state machine; all UDP calls are made by its owner worker."""

    def __init__(self, request: Callable[[str], dict[str, object]], *, mode: str = "junction"):
        if mode not in {"junction", "continuous"}:
            raise ValueError("control mode must be junction or continuous")
        self.request = request
        self.mode = mode
        self.snapshot = ControlSnapshot()
        self.events: list[dict[str, object]] = []
        self._window_id = 0
        self._window_started = 0.0
        self._visit_number = 0
        self._key: tuple[int, ...] | None = None
        self._token: tuple[int, ...] | None = None
        self._consumed = False
        self._owns_pause = False
        self._pause_key: tuple[int, ...] | None = None
        self._suspended = False
        self._needs_cancel = True
        self.allow_motion: Callable[[], bool] = lambda: True

    @staticmethod
    def _junction_key(nav: dict[str, object]) -> tuple[int, ...]:
        return (int(nav.get("maze_generation", 0)), *(int(nav[key]) for key in ("x", "y", "heading", "steps")))

    def _expected(self, nav: dict[str, object]) -> dict[str, object] | None:
        if self.mode == "continuous" and "maze_generation" not in nav:
            return None
        return {key: nav[key] for key in ("maze_generation", "x", "y", "heading", "steps", "paused", "state", "pending_turn") if key in nav}

    def _claim_pause(self, nav: dict[str, object]) -> None:
        if nav["paused"] is not True:
            raise ValueError("Unity pause ACK did not pause navigation")
        self._owns_pause = True
        self._pause_key = self._junction_key(nav)

    def _set(self, phase: str, nav: dict[str, object], error: str = "") -> ControlSnapshot:
        self.snapshot = ControlSnapshot(phase, self._window_id, self._token, dict(nav), error, self._window_started)
        return self.snapshot

    def _new_window(self) -> None:
        self._window_id += 1
        self._window_started = time.monotonic()

    def _send(self, command: str, *, expected: dict[str, object] | None = None) -> dict[str, object]:
        if command in {"left", "right", "uturn", "forward", "resume"} and not self.allow_motion():
            raise ValueError("EEG stream changed before game command; waiting for a fresh window")
        nav = validated_navigation(self.request(command, expected=expected) if expected is not None else self.request(command))
        self.events.append({"time_monotonic": time.monotonic(), "command": command, "confirmed": True, "navigation": nav})
        return nav

    def step(self, stream_ready: bool, intent: tuple[tuple[int, ...], str] | None = None) -> ControlSnapshot:
        try:
            return self._step(stream_ready, intent)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.events.append({"time_monotonic": time.monotonic(), "confirmed": False, "error": str(error)})
            if self.snapshot.phase != "unavailable":
                self._new_window()
            self._key = self._token = None
            self._needs_cancel = True
            return self._set("unavailable", {}, str(error))

    def _step(self, stream_ready: bool, intent: tuple[tuple[int, ...], str] | None) -> ControlSnapshot:
        nav = validated_navigation(self.request("status"))
        if self.mode == "junction" and type(nav.get("maze_generation")) is not int:
            raise ValueError("Junction control requires updated Unity with maze_generation and conditional commands")
        if self._owns_pause and (not nav["paused"] or self._junction_key(nav) != self._pause_key):
            self._owns_pause = False
        if nav["state"] == "Complete":
            self._token = self._key = None
            self._owns_pause = False
            return self._set("complete", nav)
        if self._needs_cancel:
            nav = self._send("cancel")
            self._needs_cancel = False
        if not stream_ready:
            if not self._suspended:
                self._new_window()
                self._key = self._token = None
                nav = self._send("cancel")
                self._suspended = True
            if not nav["paused"]:
                nav = self._send("pause", expected=self._expected(nav))
                self._claim_pause(nav)
            return self._set("signal_wait", nav)
        if self._suspended:
            self._suspended = False
            if self._owns_pause and (nav["state"] != "Junction" or self.mode == "continuous"):
                nav = self._send("resume", expected=self._expected(nav))
                self._owns_pause = False
        if nav["paused"] and not self._owns_pause:
            self._key = self._token = None
            return self._set("manual_pause", nav)
        if self.mode == "continuous":
            self._token = (self._window_id,)
            if intent and intent[0] == self._token:
                command = intent[1]
                if command not in {"left", "right", "uturn", "forward"}:
                    raise ValueError("Unsupported game command")
                nav = self._send(command, expected=self._expected(nav))
            return self._set("continuous", nav)
        if nav["state"] != "Junction":
            self._key = self._token = None
            self._consumed = False
            if self._owns_pause:
                nav = self._send("cancel")
                nav = self._send("resume", expected=self._expected(nav))
                self._owns_pause = False
            return self._set("travelling", nav)
        key = self._junction_key(nav)
        if key != self._key:
            self._key = key
            self._visit_number += 1
            self._new_window()
            self._token = (self._visit_number, *key)
            self._consumed = False
        if self._consumed:
            return self._set("submitted", nav)
        if not nav["paused"]:
            nav = self._send("pause", expected=self._expected(nav))
            self._claim_pause(nav)
            if nav["state"] != "Junction" or self._junction_key(nav) != key or not nav["paused"]:
                # A short countdown may have expired while the pause was in flight.
                self._key = self._token = None
                nav = self._send("cancel")
                raise ValueError("Unity pause ACK changed the active junction")
            # Even at the same cell, an external resume invalidates every sample and
            # queued intention from the preceding pause interval.
            self._visit_number += 1
            self._new_window()
            self._token = (self._visit_number, *key)
        if intent and intent[0] == self._token:
            command = intent[1]
            if command not in {"left", "right", "uturn", "forward"}:
                raise ValueError("Unsupported game command")
            if not nav[f"can_{command}"]:
                return self._set("waiting", nav, f"{command} exit is closed")
            self._consumed = True
            # Mark consumed before sending: an uncertain UDP ACK is never retried as a second turn.
            result = self._send(command, expected=self._expected(nav))
            valid_visit = self._junction_key(result) == key
            if command == "forward":
                valid_result = valid_visit and result["state"] == "Moving" and not result["paused"] and result.get("pending_turn") == ""
            else:
                valid_result = valid_visit and result["state"] == "Junction" and result["paused"] and result.get("pending_turn") == command
            if not valid_result:
                self._send("cancel")
                raise ValueError("Unity command ACK changed the active junction; reservation cancelled")
            if command != "forward":
                result = self._send("resume", expected=self._expected(result))
            self._owns_pause = False
            return self._set("submitted", result)
        return self._set("waiting", nav)

    def close(self) -> None:
        """Cancel our reservations and release only pauses acquired by this controller."""
        try:
            self.allow_motion = lambda: True
            nav = self._send("cancel")
            if self._owns_pause and nav["state"] != "Complete" and self._junction_key(nav) == self._pause_key:
                self._send("resume", expected=self._expected(nav))
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.events.append({"confirmed": False, "error": f"cleanup: {error}"})
        self._owns_pause = False


class GameControlWorker:
    """Polling and command ACKs never block the EEG read/classification thread."""

    def __init__(self, host: str, port: int, *, mode: str = "junction", poll_seconds: float = 0.1):
        self.controller = JunctionController(MazeUdpClient(host, port).request, mode=mode)
        self.poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = False
        self._revision = 0
        self._reset_required = True
        self._intent: tuple[tuple[int, ...], str] | None = None
        self._snapshot = ControlSnapshot()
        self._events: list[dict[str, object]] = []
        self._thread: threading.Thread | None = None

    @property
    def snapshot(self) -> ControlSnapshot:
        with self._lock:
            return self._snapshot

    def set_stream_ready(self, ready: bool) -> None:
        with self._lock:
            self._ready = bool(ready)
            if not ready:
                self._intent = None
                self._revision += 1
                self._reset_required = True

    def submit(self, command: str, visit_token: tuple[int, ...] | None) -> bool:
        with self._lock:
            if not self._ready or self._reset_required or not self._snapshot.accepting or visit_token != self._snapshot.visit_token or self._intent is not None:
                return False
            self._intent = (visit_token, command)
            return True

    def drain_events(self) -> list[dict[str, object]]:
        with self._lock:
            events, self._events = self._events, []
            return events

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="Unity BCI control", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    ready, intent = self._ready, self._intent
                    self._intent = None
                    revision = self._revision
                    if self._reset_required:
                        ready, intent = False, None
                        self._reset_required = False
                def allow_motion():
                    with self._lock:
                        return self._ready and self._revision == revision and not self._stop.is_set()
                self.controller.allow_motion = allow_motion
                snapshot = self.controller.step(ready, intent)
                with self._lock:
                    self._snapshot = snapshot
                    self._events.extend(self.controller.events)
                    self.controller.events.clear()
                self._stop.wait(self.poll_seconds)
        finally:
            self.controller.close()
            with self._lock:
                self._events.extend(self.controller.events)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("Unity control worker did not stop within its network deadline")


def model_control_provenance(bundle: object, *, allow_unvalidated: bool = False) -> dict[str, object]:
    provenance = dict(getattr(bundle, "provenance", {}) or {})
    validated = (
        provenance.get("data_origin") in {"hardware_recorded", "lsl_recorded"}
        and provenance.get("calibration_passed") is True
    )
    if not validated and not allow_unvalidated:
        raise ValueError("Unity control requires recorded EEG with passed calibration; use --allow-unvalidated-control for an explicit software validation run")
    return {**provenance, "control_validated": validated, "unvalidated_control_override": bool(allow_unvalidated and not validated)}

"""Thread-safe dashboard state, signal monitoring and owned workflow processes."""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from bci_maze.lk_mini.cli import add_source_arguments, source_from_args
from bci_maze.lk_mini.game_control import MazeUdpClient
from bci_maze.lk_mini.quality import assess_signal_quality
from bci_maze.lk_mini.source import DEFAULT_CHANNEL_NAMES
from .experiments import KINDS as EXPERIMENT_KINDS, FIELDS as EXPERIMENT_FIELDS, build_experiment_command


ROOT = Path(__file__).resolve().parents[3]
SOURCE_KEYS = {"backend", "ip_address", "ip_port", "sample_rate", "lsl_name", "replay", "recovery_profile"}
JOB_KEYS = SOURCE_KEYS | {"kind", "subject", "session", "preset", "recordings", "model", "unity_control", "unity_host", "unity_port"} | EXPERIMENT_FIELDS
DEFAULTS = dict(backend="brainflow", ip_address="192.168.4.1", ip_port=12345,
                sample_rate=250, unity_host="127.0.0.1", unity_port=7777, recovery_profile="standard")
RECOVERY_PROFILES = {
    "standard": dict(connect_timeout=12.0, no_data_timeout=3.0, shutdown_timeout=4.0,
                     retry_delay=0.5, max_retry_delay=3.0, device_timeout=5),
    "fast": dict(connect_timeout=8.0, no_data_timeout=1.5, shutdown_timeout=1.5,
                 retry_delay=0.2, max_retry_delay=1.0, device_timeout=2),
}
WORKFLOWS = {"collect": "collect_lk_mi.py", "train": "train_lk_mi.py",
             "finetune": "finetune_lk_semhaf.py", "online": "online_lk_mi.py"}


class ConflictError(RuntimeError):
    """A requested operation conflicts with an active operation."""


def _keys(value: dict, allowed: set[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError("请求内容必须是 JSON 对象。")
    extra = set(value) - allowed
    if extra:
        raise ValueError("未知参数：" + ", ".join(sorted(extra)))


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} 必须为 {low}–{high} 范围内的整数。")
    return value


def _host(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]{0,252}", value):
        raise ValueError("地址须使用 IPv4 地址或主机名。")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,63}", value):
        raise ValueError(f"{field} 请使用 1–64 位英文字母、数字、下划线或连字符。")
    return value


def _idle_monitor() -> dict:
    return dict(state="idle", backend="", sample_rate=250, channel_names=list(DEFAULT_CHANNEL_NAMES),
                samples=0, elapsed=0.0, last_sample_age=None, segment_id=0, recovery_stats={},
                source_connection_state="stopped", recovery_events=[], connection_parameters={},
                recovery_profile="standard",
                error="", waveforms=[], quality=[], window_seconds=5.0, waveform_seconds=0.0, stop_requested=False)


def _idle_job() -> dict:
    return dict(state="idle", kind="", command=[], logs=[], exit_code=None, output="", error="")


class DashboardController:
    def __init__(self, root: Path = ROOT, *, source_factory=source_from_args):
        self.root = Path(root).resolve()
        self._source_factory = source_factory
        self._lock = threading.RLock()
        self._monitor = _idle_monitor()
        self._job = _idle_job()
        self._unity = dict(connected=False, navigation={}, error="", updated_at=None)
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_started = self._monitor_last = 0.0
        self._process: subprocess.Popen | None = None
        self._job_thread: threading.Thread | None = None
        self._stop_file: Path | None = None
        self._logs: deque[str] = deque(maxlen=300)
        self._closed = False

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _input_path(self, value: object, category: str) -> Path:
        if not isinstance(value, str) or not value.strip() or len(value) > 500:
            raise ValueError("请选择已有记录或模型文件。")
        candidate = (self.root / value).resolve()
        locations = [self.root / "data/recordings"] if category == "recording" else [
            self.root / "outputs/subject_models", self.root / "outputs/pretrained"]
        suffixes = {".npz"} if category == "recording" else {".pt", ".joblib"}
        if not any(candidate.is_relative_to(location) for location in locations):
            raise ValueError("文件须位于项目的记录或模型目录。")
        if not candidate.is_relative_to(self.root) or candidate.suffix.lower() not in suffixes or not candidate.is_file():
            raise ValueError("所选记录或模型文件不存在，或格式不受支持。")
        return candidate

    def config(self) -> dict:
        def available(category: str) -> list[str]:
            folders = ["data/recordings"] if category == "recording" else ["outputs/subject_models", "outputs/pretrained"]
            patterns = ["*.npz"] if category == "recording" else ["*.pt", "*.joblib"]
            found = []
            for folder in folders:
                for pattern in patterns:
                    for path in (self.root / folder).rglob(pattern):
                        try:
                            self._input_path(self._relative(path), category)
                        except (ValueError, OSError):
                            continue
                        found.append(self._relative(path))
            return sorted(found)
        return dict(defaults=dict(DEFAULTS), recordings=available("recording"),
                    models=available("model"), channels=list(DEFAULT_CHANNEL_NAMES),
                    recovery_profiles=deepcopy(RECOVERY_PROFILES))

    def snapshot(self) -> dict:
        with self._lock:
            monitor = deepcopy(self._monitor)
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                monitor["elapsed"] = round(time.monotonic() - self._monitor_started, 2)
            if self._monitor_last:
                monitor["last_sample_age"] = round(time.monotonic() - self._monitor_last, 3)
                if monitor["last_sample_age"] > 1.5 and monitor["state"] == "streaming":
                    monitor.update(state="recovering", quality=[], waveforms=[], waveform_seconds=0.0)
            job = deepcopy(self._job)
            job["logs"] = list(self._logs)
            job["output_exists"] = bool(job["output"] and (self.root / job["output"]).is_file())
            return dict(monitor=monitor, job=job, unity=deepcopy(self._unity))

    def _busy(self) -> None:
        if self._closed:
            raise ConflictError("控制台正在关闭。")
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            raise ConflictError("请先停止信号监测，等待设备释放后再启动任务。")
        if self._process is not None and self._process.poll() is None:
            raise ConflictError("已有任务运行中，请等待任务结束或先停止任务。")
        if self._job_thread is not None and self._job_thread.is_alive():
            raise ConflictError("上一任务正在整理输出，请稍后再启动。")

    def _source_arguments(self, payload: dict) -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
        add_source_arguments(parser)
        args = parser.parse_args([])
        args.backend = payload.get("backend", DEFAULTS["backend"])
        if not isinstance(args.backend, str) or args.backend not in {"brainflow", "lsl", "replay", "synthetic"}:
            raise ValueError("数据源须为 brainflow、lsl、replay 或 synthetic。")
        args.sample_rate = _integer(payload.get("sample_rate", 250), "采样率", 250, 1000)
        if args.sample_rate not in {250, 500, 1000}:
            raise ValueError("采样率须为 250、500 或 1000 Hz。")
        args.ip_address = _host(payload.get("ip_address", DEFAULTS["ip_address"]))
        args.ip_port = _integer(payload.get("ip_port", 12345), "设备端口", 1, 65535)
        args.lsl_name = payload.get("lsl_name") or None
        if args.lsl_name is not None and (not isinstance(args.lsl_name, str) or len(args.lsl_name) > 200):
            raise ValueError("LSL 流名称须为最多 200 字的文本。")
        args.replay = self._input_path(payload.get("replay"), "recording") if args.backend == "replay" else None
        args.recovery_profile = payload.get("recovery_profile", "standard")
        if not isinstance(args.recovery_profile, str) or args.recovery_profile not in RECOVERY_PROFILES:
            raise ValueError("恢复配置须为 standard 或 fast。")
        for name, value in RECOVERY_PROFILES[args.recovery_profile].items():
            setattr(args, name, value)
        return args

    @staticmethod
    def _connection_parameters(args: argparse.Namespace) -> dict:
        if args.backend != "brainflow":
            return {"auto_reconnect": False}
        return dict(auto_reconnect=args.auto_reconnect, **{
            name: getattr(args, name) for name in RECOVERY_PROFILES[args.recovery_profile]
        })

    @staticmethod
    def _source_telemetry(source) -> dict:
        return dict(source_connection_state=str(source.connection_state),
                    recovery_stats=dict(source.recovery_stats),
                    recovery_events=deepcopy(list(source.recovery_events)[-50:]))

    def start_monitor(self, payload: dict) -> dict:
        _keys(payload, SOURCE_KEYS)
        args = self._source_arguments(payload)
        with self._lock:
            self._busy()
            self._monitor = _idle_monitor()
            self._monitor.update(state="connecting", backend=args.backend, sample_rate=args.sample_rate,
                                 source_connection_state="connecting", recovery_profile=args.recovery_profile,
                                 connection_parameters=self._connection_parameters(args))
            self._monitor_started, self._monitor_last = time.monotonic(), 0.0
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_run, args=(args,),
                                                    name="Dashboard EEG monitor", daemon=True)
            self._monitor_thread.start()
        return self.snapshot()

    def _monitor_run(self, args: argparse.Namespace) -> None:
        source = None
        try:
            source = self._source_factory(args)
            source.start()
            rate, names = int(source.sample_rate), tuple(source.channel_names)
            if rate <= 0 or not names:
                raise ValueError("数据源未提供有效通道和采样率。")
            buffer = np.empty((len(names), 0), dtype=np.float64)
            segment, next_quality = 0, 0.0
            with self._lock:
                self._monitor.update(sample_rate=rate, channel_names=list(names), **self._source_telemetry(source))
            while not self._monitor_stop.is_set():
                chunk = source.read(timeout=0.25)
                now = time.monotonic()
                recovering = bool(source.recovering)
                stale = bool(self._monitor_last and now - self._monitor_last > 1.5)
                boundary = chunk.segment_id != segment or recovering or stale
                if boundary:
                    buffer = np.empty((len(names), 0), dtype=np.float64)
                    segment = int(chunk.segment_id)
                    next_quality = 0.0
                    with self._lock:
                        self._monitor.update(quality=[], waveforms=[], waveform_seconds=0.0)
                with self._lock:
                    self._monitor.update(segment_id=segment, **self._source_telemetry(source))
                if not chunk.data.shape[1]:
                    with self._lock:
                        if recovering or stale or not self._monitor["samples"]:
                            self._monitor["state"] = "recovering" if self._monitor["samples"] else "connecting"
                    if source.exhausted:
                        break
                    self._monitor_stop.wait(0.01)
                    continue
                if chunk.data.shape[0] != len(names):
                    raise ValueError("数据通道数发生变化，请重新启动监测。")
                buffer = np.concatenate((buffer, chunk.data), axis=1)[:, -rate * 5:]
                indices = np.linspace(0, buffer.shape[1] - 1, min(1250, buffer.shape[1]), dtype=int)
                waveforms = np.nan_to_num(buffer[:, indices], nan=0.0, posinf=0.0, neginf=0.0).round(3).tolist()
                quality = None
                if now >= next_quality and buffer.shape[1] >= rate * 2:
                    quality = [dict(name=name, **item.to_dict()) for name, item in
                               zip(names, assess_signal_quality(buffer, rate))]
                    next_quality = now + 0.5
                with self._lock:
                    self._monitor_last = now
                    self._monitor.update(state="streaming", waveforms=waveforms, waveform_seconds=buffer.shape[1] / rate,
                                         samples=self._monitor["samples"] + int(chunk.data.shape[1]))
                    if quality is not None:
                        self._monitor["quality"] = quality
        except Exception as error:
            with self._lock:
                self._monitor.update(state="error", error=f"{type(error).__name__}: {error}", quality=[], waveforms=[])
        finally:
            if source is not None:
                try:
                    source.stop()
                except Exception as error:
                    with self._lock:
                        self._monitor.update(state="error", error=f"关闭数据源失败：{error}")
                finally:
                    with self._lock:
                        self._monitor.update(**self._source_telemetry(source))
            with self._lock:
                self._monitor["elapsed"] = round(time.monotonic() - self._monitor_started, 2)
                if self._monitor["state"] != "error":
                    self._monitor["state"] = "stopped"
                self._monitor.update(quality=[], waveforms=[], waveform_seconds=0.0, stop_requested=False)

    def stop_monitor(self) -> dict:
        with self._lock:
            self._monitor_stop.set()
            self._monitor["stop_requested"] = bool(self._monitor_thread and self._monitor_thread.is_alive())
        return self.snapshot()

    def _python(self, torch: bool) -> str:
        environment = ".venv" if torch else ".venv-mi"
        executable = self.root / environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not executable.is_file():
            raise ValueError(f"缺少 {environment} Python 环境，请按项目运行说明配置。")
        return str(executable)

    def _source_cli(self, payload: dict) -> list[str]:
        args = self._source_arguments(payload)
        flags = ["--backend", args.backend, "--sample-rate", str(args.sample_rate),
                 "--ip-address", args.ip_address, "--ip-port", str(args.ip_port)]
        for name in RECOVERY_PROFILES[args.recovery_profile]:
            flags += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.lsl_name:
            flags += ["--lsl-name", args.lsl_name]
        if args.replay:
            flags += ["--replay", str(args.replay)]
        return flags

    def _build_job(self, payload: dict) -> tuple[list[str], Path]:
        _keys(payload, JOB_KEYS)
        kind = payload.get("kind")
        if not isinstance(kind, str) or kind not in set(WORKFLOWS) | EXPERIMENT_KINDS:
            raise ValueError("请选择采集、训练、在线识别或子实验任务。")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        if kind in EXPERIMENT_KINDS:
            return build_experiment_command(self, payload, stamp)
        arguments: list[str] = []
        torch = kind == "finetune"
        if kind in {"collect", "online"}:
            arguments += self._source_cli(payload)
        if kind == "collect":
            subject = _identifier(payload.get("subject"), "受试者编号")
            session = _identifier(payload.get("session") or "calibration", "会话编号")
            preset = payload.get("preset", "short")
            if not isinstance(preset, str) or preset not in {"short", "standard", "full"}:
                raise ValueError("采集方案须为 short、standard 或 full。")
            output = self.root / "data/recordings" / subject / f"{session}-{stamp}.npz"
            arguments += ["--subject", subject, "--session", session, "--preset", preset, "--output", str(output)]
        elif kind in {"train", "finetune"}:
            subject = _identifier(payload.get("subject"), "受试者编号")
            recordings = payload.get("recordings")
            if not isinstance(recordings, list) or not 1 <= len(recordings) <= 100:
                raise ValueError("请选择 1–100 个采集记录。")
            arguments += [str(self._input_path(value, "recording")) for value in recordings]
            suffix = ".pt" if torch else ".joblib"
            output = self.root / "outputs/subject_models" / subject / f"{kind}-{stamp}{suffix}"
            arguments += ["--subject", subject, "--output", str(output)]
            if kind == "finetune":
                pretrained = payload.get("model") or "outputs/pretrained/lk_semhaf16.pt"
                selected = self._input_path(pretrained, "model")
                if selected.suffix.lower() != ".pt":
                    raise ValueError("SE-MHAF 微调需要 .pt 预训练模型。")
                arguments += ["--pretrained", str(selected)]
        else:
            model = self._input_path(payload.get("model"), "model")
            torch = model.suffix.lower() == ".pt"
            output = self.root / "outputs/online" / f"dashboard-{stamp}.jsonl"
            arguments += ["--model", str(model), "--plot", "--log", str(output)]
            control = payload.get("unity_control", False)
            if type(control) is not bool:
                raise ValueError("unity_control 必须为布尔值。")
            if control:
                arguments += ["--unity-host", _host(payload.get("unity_host", "127.0.0.1")),
                              "--unity-port", str(_integer(payload.get("unity_port", 7777), "Unity 端口", 1, 65535))]
        if not output.resolve().is_relative_to(self.root):
            raise ValueError("输出目录须位于当前项目中。")
        executable = self._python(torch)
        script = self.root / "scripts" / WORKFLOWS[kind]
        if not script.is_file():
            raise ValueError(f"任务脚本不存在：{script.name}")
        return [executable, "-X", "utf8", "-u", str(script), *arguments], output

    def start_job(self, payload: dict) -> dict:
        command, output = self._build_job(payload)
        with self._lock:
            self._busy()
            control_folder = self.root / "tmp/dashboard"
            if not control_folder.resolve().is_relative_to(self.root):
                raise ValueError("任务控制目录须位于当前项目中。")
            control_folder.mkdir(parents=True, exist_ok=True)
            self._stop_file = control_folder / f"stop-{uuid.uuid4().hex}.signal"
            worker_command = [*command[:4], "-m", "bci_maze.dashboard.job_worker",
                              str(self._stop_file), command[4], *command[5:]]
            environment = os.environ.copy()
            environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(self.root / "src"))
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._process = subprocess.Popen(worker_command, cwd=self.root, env=environment,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace", bufsize=1,
                                             creationflags=creationflags)
            self._logs.clear()
            self._job = dict(state="running", kind=payload["kind"], command=command, logs=[],
                             exit_code=None, output=self._relative(output), error="", experiment=payload.get("experiment"))
            self._job_thread = threading.Thread(target=self._read_job, args=(self._process, self._stop_file),
                                                name="Dashboard workflow log", daemon=True)
            self._job_thread.start()
        return self.snapshot()

    def _read_job(self, process: subprocess.Popen, stop_file: Path) -> None:
        try:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                with self._lock:
                    self._logs.append(line.rstrip("\r\n")[:2000])
            code = process.wait()
            with self._lock:
                stopped = self._job["state"] == "stopping" or code == 130
                self._job.update(state="stopped" if stopped else "completed" if code == 0 else "failed", exit_code=code)
                if stopped:
                    self._job["error"] = ""
                if code and not stopped:
                    self._job["error"] = f"任务退出码 {code}，请查看运行日志。"
        finally:
            if process.stdout:
                process.stdout.close()
            stop_file.unlink(missing_ok=True)

    def stop_job(self) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None and self._stop_file:
                self._stop_file.touch()
                self._job["state"] = "stopping"
                self._job["error"] = "已请求停止，等待任务保存已完成记录并释放数据源。"
        return self.snapshot()

    def unity_request(self, payload: dict) -> dict:
        _keys(payload, {"command", "host", "port"})
        command = payload.get("command")
        if not isinstance(command, str) or command not in {"status", "pause", "resume", "left", "right", "uturn"}:
            raise ValueError("不支持此 Unity 指令。")
        host = _host(payload.get("host", "127.0.0.1"))
        port = _integer(payload.get("port", 7777), "Unity 端口", 1, 65535)
        with self._lock:
            if self._process is not None and self._process.poll() is None and self._job["kind"] == "online" and command not in {"status", "pause"}:
                raise ConflictError("在线识别运行时可查看状态或暂停游戏；其他手动控制请先停止识别。")
        try:
            navigation = MazeUdpClient(host, port, timeout=0.5).request(command)
            result = dict(connected=True, navigation=navigation, error="", updated_at=time.time())
        except Exception as error:
            result = dict(connected=False, navigation={}, error=f"{type(error).__name__}: {error}", updated_at=time.time())
        with self._lock:
            self._unity = result
        return self.snapshot()

    def close(self, timeout: float = 15.0) -> bool:
        with self._lock:
            self._closed = True
        self.stop_monitor()
        self.stop_job()
        deadline = time.monotonic() + timeout
        for thread in (self._monitor_thread, self._job_thread):
            if thread is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        return not ((self._monitor_thread and self._monitor_thread.is_alive()) or
                    (self._process and self._process.poll() is None))

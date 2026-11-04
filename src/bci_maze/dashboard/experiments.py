"""Validated experiment commands and report access for the local dashboard."""
from __future__ import annotations

import json
from pathlib import Path


KINDS = {"experiment_collect", "experiment_analyze", "experiment_demo"}
FIELDS = {"experiment", "trials_per_class", "blocks", "permutations", "split", "wear_state"}


def build_experiment_command(controller, payload, stamp):
    from .controller import _identifier, _integer

    experiment = payload.get("experiment", "imagery")
    if not isinstance(experiment, str) or experiment not in {"motor", "imagery", "benchmark"}:
        raise ValueError("子实验须为 motor、imagery 或 benchmark。")
    action = payload["kind"].removeprefix("experiment_")
    if action == "collect" and experiment == "benchmark":
        raise ValueError("基线模型验证需要选择已采集的运动想象记录。")
    subject = _identifier(payload.get("subject", "DEMO_EXPERIMENT" if action == "demo" else ""), "受试者编号")
    session = _identifier(payload.get("session") or f"{experiment}-{stamp}", "会话编号")
    permutations = _integer(payload.get("permutations", 30), "随机标签次数", 5, 200)
    split = payload.get("split", "auto")
    if not isinstance(split, str) or split not in {"auto", "session", "block"}:
        raise ValueError("验证划分须为 auto、session 或 block。")
    output_dir = controller.root / "outputs/experiments" / subject / f"{experiment}-{stamp}"
    recording = controller.root / "data/recordings" / subject / f"experiment-{experiment}-{stamp}.npz"
    for path in (output_dir, recording):
        if not path.resolve().is_relative_to(controller.root):
            raise ValueError("实验输出须位于当前项目中。")
    args = ["--action", action, "--experiment", experiment, "--subject", subject,
            "--session", session, "--output-dir", str(output_dir),
            "--recording-output", str(recording), "--permutations", str(permutations), "--split", split]
    if action == "analyze":
        recordings = payload.get("recordings")
        if not isinstance(recordings, list) or not 1 <= len(recordings) <= 100:
            raise ValueError("请先选择 1–100 个同一受试者的采集记录。")
        paths = [controller._input_path(value, "recording") for value in recordings]
        if len(set(paths)) != len(paths):
            raise ValueError("实验记录不能重复选择。")
        args += ["--recordings", *map(str, paths)]
    else:
        trials = _integer(payload.get("trials_per_class", 10), "每类试次数", 4, 80)
        blocks = _integer(payload.get("blocks", 2), "组数", 2, 8)
        if trials % blocks or trials < blocks:
            raise ValueError("每类试次数须为组数的整数倍，每组包含左右两类。")
        args += ["--trials-per-class", str(trials), "--blocks", str(blocks)]
        if action == "collect":
            wear_state = payload.get("wear_state", "unknown")
            if not isinstance(wear_state, str) or wear_state not in {"unknown", "worn", "unworn"}:
                raise ValueError("佩戴状态须为 unknown、worn 或 unworn。")
            args += ["--wear-state", wear_state]
            source_args = controller._source_arguments(payload)
            if source_args.backend == "replay":
                raise ValueError("已录制数据请使用“分析所选记录”；新提示采集支持 BrainFlow、LSL 或仿真数据源。")
            args += controller._source_cli(payload)
    script = controller.root / "scripts/run_lk_experiment.py"
    if not script.is_file():
        raise ValueError("子实验脚本尚未安装。")
    command = [controller._python(False), "-X", "utf8", "-u", str(script), *args]
    return command, output_dir / "report.json"


def report_path(root: Path, value: str, *, asset=False) -> Path:
    if not isinstance(value, str) or not value.strip() or len(value) > 800:
        raise ValueError("请选择实验结果文件。")
    destination = (root / value).resolve()
    allowed = (root / "outputs/experiments").resolve()
    if not allowed.is_relative_to(root.resolve()) or not destination.is_relative_to(allowed):
        raise ValueError("结果文件须位于本项目实验输出目录。")
    if asset:
        if destination.suffix.lower() != ".png":
            raise ValueError("实验图表使用 PNG 格式。")
    elif destination.name != "report.json":
        raise ValueError("请选择实验 report.json。")
    if not destination.is_file():
        raise ValueError("实验结果尚未生成或文件不存在。")
    return destination


def list_reports(root: Path) -> list[dict]:
    results = []
    for path in (root / "outputs/experiments").glob("*/*/report.json"):
        try:
            relative = path.relative_to(root).as_posix()
            validated = report_path(root, relative)
            value = json.loads(validated.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                continue
            results.append(dict(path=relative, title=f"{value.get('subject', '?')} · {value.get('task', '?')} · {path.parent.name}",
                                subject=value.get("subject"), task=value.get("task"), source_kind=value.get("source_kind")))
        except (ValueError, OSError):
            continue
    return list(reversed(sorted(results, key=lambda item: item["path"])))

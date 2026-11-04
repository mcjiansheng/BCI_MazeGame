"""Collect, analyze or simulate explicitly separated MindMaze subexperiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bci_maze.lk_mini.cli import add_source_arguments
from bci_maze.experiments.collection import (
    TASKS, collect_experiment, create_demo_recording, emit_progress,
)


def validate_recordings(paths, *, subject: str, experiment: str) -> list[Path]:
    if not paths:
        raise ValueError("请选择至少一个已有实验记录。")
    selected = [Path(path).resolve() for path in paths]
    if len(set(selected)) != len(selected):
        raise ValueError("分析记录不能重复。")
    for path in selected:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
        if not isinstance(metadata, dict) or not isinstance(metadata.get("configuration", {}), dict):
            raise ValueError(f"记录 {path.name} 的元数据格式错误。")
        if metadata.get("subject_id") != subject:
            raise ValueError(f"记录 {path.name} 的受试者 {metadata.get('subject_id')} 与 {subject} 不符。")
        task = metadata.get("configuration", {}).get("task", metadata.get("task", "unspecified"))
        if task not in {"unspecified", TASKS[experiment]}:
            raise ValueError(f"记录 {path.name} 的任务 {task} 与所选 {TASKS[experiment]} 不符。")
    return selected


def write_empty_report(args, result) -> Path:
    report = dict(report_version=1, subject=args.subject, task=TASKS[args.experiment],
                  status="no_complete_trials", source_kind={"brainflow": "hardware", "lsl": "lsl",
                  "synthetic": "synthetic", "replay": "replay"}.get(args.backend, "unspecified"),
                  sample_rate=args.sample_rate, channels=list(args.channel_names),
                  classes=["left_hand", "right_hand"], n_trials=0,
                  class_counts={"left_hand": 0, "right_hand": 0}, figures=[], analysis={}, benchmark=None,
                  baseline_available=False, notes=["未取得完整且同段的基线与任务试次，请重新采集。"],
                  completed_trials=0, discarded_trials=result.discarded,
                  capture_interrupted=result.interrupted, capture_error=result.error,
                  message="未取得完整且同段的基线与任务试次，请重新采集。")
    destination = args.output_dir / "report.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=args.output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def finish(report, args, result):
    """Persist collection provenance and target accounting with the analysis."""
    if result is None or result.summary is None:
        return
    recording = Path(result.path).resolve()
    try:
        recording_name = recording.relative_to(ROOT).as_posix()
    except ValueError:
        recording_name = str(recording)
    report.update(recording_path=recording_name, recording_paths=[recording_name],
                  collection_summary=result.summary, wear_state=args.wear_state,
                  capture_interrupted=result.interrupted, capture_error=result.error)
    if not result.summary["target_met"]:
        reason = {"user_interrupted": "操作员停止采集", "stable_wait_timeout": "等待稳定信号达到时限",
                  "session_duration_limit": "整场采集达到时限", "attempt_limits_reached": "已达到单个目标的尝试上限",
                  "capture_error": "采集发生错误"}.get(result.summary["stop_reason"], "采集已结束")
        report.setdefault("notes", []).append(
            f"本次目标 {result.summary['target_trials']} 个有效试次，已保留 {result.summary['usable_trials']} 个；"
            f"结束原因：{reason}。全部已收到的连续信号保存于 {recording_name}。")
    descriptor, temporary = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=args.output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, args.output_dir / "report.json")
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("collect", "analyze", "demo"), required=True)
    parser.add_argument("--experiment", choices=tuple(TASKS), required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recording-output", type=Path)
    parser.add_argument("--recordings", type=Path, nargs="+")
    parser.add_argument("--trials-per-class", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument("--split", choices=("auto", "session", "block"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--max-attempts-per-slot", type=int, default=3)
    parser.add_argument("--stable-wait-timeout", type=float, default=60.0)
    parser.add_argument("--max-duration-seconds", type=float, default=0,
                        help="0 selects max(300, planned trials * 30) seconds")
    parser.add_argument("--wear-state", choices=("unknown", "worn", "unworn"), default="unknown")
    add_source_arguments(parser)
    args = parser.parse_args(argv)
    for field in ("subject", "session"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", getattr(args, field)):
            parser.error(f"--{field} 须使用 1–64 位英文字母、数字、下划线或连字符。")
    if not 1 <= args.permutations <= 1000 or not np.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("置换次数须为 1–1000，预热时长须为非负有限数值。")
    if (not 1 <= args.max_attempts_per_slot <= 3
            or not np.isfinite([args.stable_wait_timeout, args.max_duration_seconds]).all()
            or args.stable_wait_timeout <= 0 or args.max_duration_seconds < 0):
        parser.error("每个试次最多尝试 1–3 次，等待时限须为正数，整场时限须为非负有限数。")
    if args.action == "collect" and args.experiment == "benchmark":
        parser.error("benchmark 使用 analyze 或 demo。")
    if args.action != "analyze" and args.recording_output is None:
        parser.error("collect 和 demo 需要 --recording-output。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = None
    if args.action == "analyze":
        recordings = validate_recordings(args.recordings, subject=args.subject, experiment=args.experiment)
    elif args.action == "demo":
        emit_progress("waiting", text="正在生成明确标记的合成实验记录。")
        recordings = [create_demo_recording(
            args.recording_output, experiment=args.experiment, subject=args.subject, session=args.session,
            trials_per_class=args.trials_per_class, blocks=args.blocks, seed=args.seed,
            sample_rate=args.sample_rate, channel_names=args.channel_names,
        )]
    else:
        result = collect_experiment(args)
        if result.path is None:
            destination = write_empty_report(args, result)
            emit_progress("complete", total_trials=2 * args.trials_per_class,
                          text="采集已停止，尚无完整试次；状态说明已保存。")
            print(f"实验报告：{destination}", flush=True)
            return 130 if result.interrupted else 3
        recordings = [result.path]
    from bci_maze.experiments.analysis import analyze_recordings
    planned = (2 * args.trials_per_class if args.action == "collect" else
               40 if args.action == "demo" and args.experiment == "benchmark" else
               2 * args.trials_per_class if args.action == "demo" else 0)
    summary = result.summary if result else None
    if summary:
        planned = summary["target_trials"]
    progress = (dict(attempt_index=summary["total_attempts"] - 1, total_attempts=summary["total_attempts"],
                     retained_trials=summary["usable_trials"], target_trials=summary["target_trials"],
                     discarded_trials=summary["incomplete_trials"] + summary["excluded_complete_trials"],
                     retry_count=summary["last_retry_index"]) if summary else {})
    emit_progress("waiting", trial_index=summary["last_planned_slot"] if summary else result.completed if result else 0,
                  total_trials=planned, text="正在分析已保存的实验记录。", **progress)
    report = analyze_recordings(recordings, args.output_dir, run_benchmark=args.experiment == "benchmark",
                                permutations=args.permutations, seed=args.seed, split=args.split)
    finish(report, args, result)
    retained = int(report["n_trials"])
    emit_progress("complete", trial_index=summary["last_planned_slot"] if summary else retained,
                  total_trials=planned or retained,
                  text=f"实验分析完成，保留 {retained} 次有效试次，结果已保存。", **progress)
    print(f"实验报告：{args.output_dir / 'report.json'}", flush=True)
    return 130 if result and result.interrupted else 3 if result and result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())

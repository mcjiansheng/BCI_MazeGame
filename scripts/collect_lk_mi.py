"""Run a visual motor-imagery calibration experiment and save labeled EEG."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from bci_maze.lk_mini.cli import add_source_arguments, source_from_args
from bci_maze.lk_mini.quality import assess_signal_quality
from bci_maze.lk_mini.recording import (
    ContinuousRecorder,
    TrialRecord,
    save_mi_recording,
)
from bci_maze.lk_mini.simulation import save_synthetic_mi_recording


INSTRUCTIONS = {
    "left_hand": ("←", "想象左手反复握紧和放松\n保持身体完全不动"),
    "right_hand": ("→", "想象右手反复握紧和放松\n保持身体完全不动"),
    "feet": ("↑", "想象双脚同时踩踏或脚趾运动\n不要真的移动双脚"),
    "tongue": ("●", "想象舌头反复活动\n不要真的移动舌头或下颌"),
}

COLLECTION_PRESETS = {
    "short": (10, 2),
    "standard": (20, 4),
    "full": (40, 4),
}
PRESET_NAMES = {"short": "短轮", "standard": "标准", "full": "完整"}
SOURCE_NAMES = {"brainflow": "设备数据", "lsl": "LSL 数据", "replay": "回放数据", "synthetic": "合成数据"}


class CueDisplay:
    """Tk-based participant display; press A to mark a trial, Esc to stop safely."""

    def __init__(self, fullscreen: bool):
        try:
            import tkinter as tk
        except ImportError as exception:
            raise RuntimeError("Tkinter is required for the MI cue window") from exception
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("LK-Mini 运动想象校准")
        self.root.configure(bg="#10141d")
        self.root.geometry("1100x720")
        self.root.minsize(900, 700)
        if fullscreen:
            self.root.attributes("-fullscreen", True)
        self.plan_label = tk.Label(self.root, text="", font=("Microsoft YaHei", 12), fg="#aab4c8", bg="#10141d", wraplength=850)
        self.plan_label.pack(pady=(12, 0))
        self.phase = tk.Label(self.root, text="准备", font=("Microsoft YaHei", 32), fg="#aab4c8", bg="#10141d")
        self.phase.pack(pady=(18, 10))
        self.symbol = tk.Label(self.root, text="+", font=("Arial", 150, "bold"), fg="white", bg="#10141d")
        self.symbol.pack(pady=10)
        self.instruction = tk.Label(
            self.root,
            text="请放松",
            font=("Microsoft YaHei", 28),
            justify="center",
            fg="white",
            bg="#10141d",
        )
        self.instruction.pack(pady=10)
        self.progress = tk.Label(self.root, text="", font=("Microsoft YaHei", 18), fg="#69d0ff", bg="#10141d")
        self.progress.pack(pady=20)
        self.quality = tk.Label(self.root, text="信号检查中…", font=("Microsoft YaHei", 12), fg="#8bd49c", bg="#10141d", wraplength=850)
        self.quality.pack(side="bottom", pady=15)
        self._artifact = False
        self.aborted = False
        self.root.bind("<Escape>", self._abort)
        self.root.bind("<KeyPress-a>", self._mark_artifact)
        self.root.bind("<KeyPress-A>", self._mark_artifact)
        self.root.protocol("WM_DELETE_WINDOW", self._abort)
        self.root.update()

    def _abort(self, _event=None):
        self.aborted = True

    def _mark_artifact(self, _event=None):
        self._artifact = True
        self.quality.configure(text="本 trial 已标记伪迹（A）", fg="#ff9e64")

    def begin_trial(self) -> None:
        self._artifact = False

    def set_plan(self, text: str) -> None:
        self.plan_label.configure(text=text)

    def consume_artifact(self) -> bool:
        return self._artifact

    def show(
        self,
        phase: str,
        symbol: str,
        instruction: str,
        progress: str,
        duration: float,
        quality_provider,
    ) -> bool:
        self.phase.configure(text=phase)
        self.symbol.configure(text=symbol)
        self.instruction.configure(text=instruction)
        self.progress.configure(text=progress)
        started = time.monotonic()
        next_quality = started
        while not self.aborted and time.monotonic() - started < duration:
            remaining = max(0.0, duration - (time.monotonic() - started))
            self.progress.configure(text=f"{progress}   {remaining:0.1f}s")
            if time.monotonic() >= next_quality:
                good, total, details = quality_provider()
                if not self._artifact:
                    color = "#aab4c8" if good is None else "#8bd49c" if good == total else "#ffcc66"
                    status = "信号质量：暂不可评估" if good is None else f"实时信号：{good}/{total} 通道良好"
                    self.quality.configure(
                        text=f"{status}\n{details}",
                        fg=color,
                    )
                next_quality = time.monotonic() + 0.75
            self.root.update()
            time.sleep(0.02)
        return not self.aborted

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


def make_schedule(classes: tuple[str, ...], blocks: int, trials_per_class: int, seed: int):
    if trials_per_class % blocks:
        raise ValueError("--trials-per-class must be divisible by --blocks")
    rng = random.Random(seed)
    per_block = trials_per_class // blocks
    schedule: list[tuple[int, str]] = []
    for block in range(blocks):
        block_trials = [label for label in classes for _ in range(per_block)]
        rng.shuffle(block_trials)
        schedule.extend((block, label) for label in block_trials)
    return schedule


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--subject", required=True, help="anonymous participant ID, e.g. S01")
    parser.add_argument("--session", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--classes", default="left_hand,right_hand")
    parser.add_argument("--preset", choices=tuple(COLLECTION_PRESETS), default="short", help="short: 10/class, 2 blocks; standard: 20/class, 4 blocks; full: 40/class, 4 blocks")
    parser.add_argument("--trials-per-class", type=int, default=None, help="override the selected preset")
    parser.add_argument("--blocks", type=int, default=None, help="override the selected preset")
    parser.add_argument("--prepare-seconds", type=float, default=2.0)
    parser.add_argument("--cue-seconds", type=float, default=1.0)
    parser.add_argument("--imagery-seconds", type=float, default=4.0)
    parser.add_argument("--rest-min", type=float, default=2.0)
    parser.add_argument("--rest-max", type=float, default=3.0)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--block-rest-seconds", type=float, default=30.0)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-location", default="unspecified", help="actual reference electrode location; use the cap/wiring label")
    parser.add_argument("--bias-location", default="unspecified", help="actual BIAS electrode location")
    parser.add_argument(
        "--simulate-mi",
        action="store_true",
        help="generate deterministic labeled MI data without opening the cue UI",
    )
    return parser


def apply_collection_preset(args: argparse.Namespace) -> None:
    trials, blocks = COLLECTION_PRESETS[args.preset]
    args.preset_overridden = args.trials_per_class is not None or args.blocks is not None
    if args.trials_per_class is None:
        args.trials_per_class = trials
    if args.blocks is None:
        args.blocks = blocks


def planned_timing(args: argparse.Namespace, total_trials: int) -> dict[str, float]:
    fixed = args.warmup_seconds + total_trials * (
        args.prepare_seconds + args.cue_seconds + args.imagery_seconds
    ) + (args.blocks - 1) * args.block_rest_seconds
    rng = random.Random(args.seed + 1)
    return {
        "minimum_seconds": fixed + total_trials * args.rest_min,
        "expected_seconds": fixed + total_trials * (args.rest_min + args.rest_max) / 2,
        "maximum_seconds": fixed + total_trials * args.rest_max,
        "seeded_seconds": fixed + sum(rng.uniform(args.rest_min, args.rest_max) for _ in range(total_trials)),
    }


def format_duration(seconds: float) -> str:
    minutes, remaining = divmod(int(round(seconds)), 60)
    return f"{minutes} 分 {remaining:02d} 秒"


def collection_configuration(args: argparse.Namespace, timing: dict, total_trials: int) -> dict:
    hardware = args.backend == "brainflow"
    synthetic = args.backend == "synthetic"
    config = {
        "backend": args.backend,
        "label_source": "visual_cue",
        "preset": args.preset,
        "preset_overridden": args.preset_overridden,
        "trials_per_class": args.trials_per_class,
        "blocks": args.blocks,
        "planned_trials": total_trials,
        "planned_timing": timing,
        "device_channels_1_based": [value + 1 for value in args.channels],
        "prepare_seconds": args.prepare_seconds,
        "cue_seconds": args.cue_seconds,
        "imagery_seconds": args.imagery_seconds,
        "rest_range_seconds": [args.rest_min, args.rest_max],
        "warmup_seconds": args.warmup_seconds,
        "block_rest_seconds": args.block_rest_seconds,
        "sample_rate": args.sample_rate,
        "units": "uV",
        "gain": args.gain if hardware else None,
        "reference": "SRB2" if hardware else ("synthetic" if synthetic else "unspecified"),
        "reference_location": "not_applicable" if synthetic else args.reference_location.strip() or "unspecified",
        "bias_location": "not_applicable" if synthetic else args.bias_location.strip() or "unspecified",
        "channel_settings": {
            "configured_by": "brainflow" if hardware else "external" if args.backend == "lsl" else "not_applicable",
            "input_mode": "normal" if hardware else None,
            "bias_enabled": True if hardware else None,
            "srb2_enabled": True if hardware else None,
            "srb1_enabled": False if hardware else None,
            "gain": args.gain if hardware else None,
            "physical_montage_verified": False,
        },
        "seed": args.seed,
    }
    if args.backend == "replay" and args.replay is not None:
        config["source_recording"] = str(args.replay.resolve())
        with np.load(args.replay, allow_pickle=False) as recorded:
            original = json.loads(str(recorded["metadata_json"].item())) if "metadata_json" in recorded.files else {}
        config["source_provenance"] = original.get("provenance", {})
        config["source_reference"] = original.get("reference", "unspecified")
    return config


def trial_progress(trials: list[TrialRecord], discarded: list[dict]) -> str:
    artifacts = sum(trial.artifact for trial in trials)
    return f"有效 {len(trials) - artifacts} · 伪迹 {artifacts} · 丢弃 {len(discarded)}"


def main() -> int:
    args = build_argument_parser().parse_args()
    apply_collection_preset(args)
    if args.no_realtime and not args.simulate_mi:
        raise ValueError("Visual MI collection requires real-time pacing; remove --no-realtime")
    if args.blocks <= 0 or args.trials_per_class <= 0:
        raise ValueError("--blocks and --trials-per-class must be positive")
    if (args.imagery_seconds <= 0 or args.rest_max < args.rest_min
            or any(value < 0 for value in (
                args.prepare_seconds, args.cue_seconds, args.rest_min,
                args.warmup_seconds, args.block_rest_seconds,
            ))):
        raise ValueError("Invalid phase durations or rest range")
    if len(args.channels) != len(args.channel_names):
        raise ValueError("--channels and --channel-names must contain the same number of items")

    classes = tuple(part.strip() for part in args.classes.split(",") if part.strip())
    if (
        len(classes) != 2
        or len(set(classes)) != len(classes)
        or any(label not in INSTRUCTIONS for label in classes)
    ):
        raise ValueError(
            f"Choose exactly two distinct classes from {tuple(INSTRUCTIONS)}; the trainer is binary"
        )
    schedule = make_schedule(classes, args.blocks, args.trials_per_class, args.seed)
    timing = planned_timing(args, len(schedule))
    configuration = collection_configuration(args, timing, len(schedule))
    preset_label = PRESET_NAMES[args.preset] + ("（自定义次数）" if args.preset_overridden else "")
    plan_text = (
        f"{preset_label} · {SOURCE_NAMES[args.backend]} · 每类 {args.trials_per_class} 次，"
        f"共 {len(schedule)} 次 / {args.blocks} 组 · 运动想象 {args.imagery_seconds:g} 秒\n"
        f"提示流程预计 {format_duration(timing['expected_seconds'])}"
        f"（{format_duration(timing['minimum_seconds'])}–{format_duration(timing['maximum_seconds'])}）"
    )
    print(plan_text)
    print("连接恢复期间暂停进入下一次提示；有效次数按完整采集且未标记伪迹的 trial 统计。")
    output = args.output or Path("data/recordings") / args.subject / f"{args.session}.npz"
    if args.simulate_mi:
        if args.backend != "synthetic":
            raise ValueError("--simulate-mi requires --backend synthetic")
        destination = save_synthetic_mi_recording(
            output,
            schedule=schedule,
            subject_id=args.subject,
            session_id=args.session,
            channel_names=args.channel_names,
            sample_rate=args.sample_rate,
            epoch_seconds=args.imagery_seconds,
            seed=args.seed,
            configuration=configuration,
        )
        print(
            f"Synthetic workflow data: {len(schedule)} trials saved to {destination}；"
            f"有效 {len(schedule)}，伪迹 0，丢弃 0。"
        )
        return 0
    source = source_from_args(args)
    display = CueDisplay(args.fullscreen)
    display.set_plan(plan_text)
    trials: list[TrialRecord] = []
    epochs: list[np.ndarray] = []
    labels: list[str] = []
    discarded_trials: list[dict[str, object]] = []
    active_trial: dict[str, object] | None = None
    rng = random.Random(args.seed + 1)
    actual_sample_rate = int(args.sample_rate)
    actual_channel_names = tuple(args.channel_names)
    recording_error: BaseException | None = None
    recording_traceback = None
    recorder = ContinuousRecorder(source)

    try:
        with recorder:
            actual_sample_rate = int(source.sample_rate)
            actual_channel_names = tuple(source.channel_names)
            if actual_sample_rate <= 0 or not actual_channel_names:
                raise RuntimeError("EEG source did not expose valid stream metadata")

            def quality_provider() -> tuple[int | None, int, str]:
                if recorder.error is not None:
                    raise RuntimeError("EEG recorder stopped unexpectedly") from recorder.error
                if recorder.recovering:
                    return None, len(actual_channel_names), "连接恢复中，等待连续数据"
                needed = actual_sample_rate * 2
                data = recorder.recent(needed)
                if data.shape[1] < needed:
                    return None, len(actual_channel_names), "等待完整的 2 秒信号"
                quality = assess_signal_quality(data, actual_sample_rate)
                details = "  ".join(
                    f"{name}: {item.status}" for name, item in zip(actual_channel_names, quality)
                    if item.status != "good"
                ) or "全部通道 good"
                return sum(item.status == "good" for item in quality), len(quality), details

            if not display.show(
                "设备预热",
                "+",
                "坐直、放松肩颈、注视中央\n不要说话或移动",
                f"预热与信号检查 · 计划 {len(schedule)} 次",
                args.warmup_seconds,
                quality_provider,
            ):
                return 1

            total_trials = len(schedule)
            for trial_index, (block, label) in enumerate(schedule):
                display.begin_trial()
                progress = (
                    f"第 {block + 1}/{args.blocks} 组   第 {trial_index + 1}/{total_trials} 次\n"
                    + trial_progress(trials, discarded_trials)
                )
                while recorder.recovering or recorder.recent(actual_sample_rate * 2).shape[1] < actual_sample_rate * 2:
                    if not display.show(
                        "等待数据", "+", "保持放松，连接恢复后继续下一次提示",
                        progress, 0.5, quality_provider,
                    ):
                        break
                if display.aborted:
                    break
                if trial_index and block != schedule[trial_index - 1][0]:
                    if not display.show(
                        "组间休息", "+", "停止想象，放松双手和肩颈",
                        progress, args.block_rest_seconds, quality_provider,
                    ):
                        break
                if not display.show(
                    "准备",
                    "+",
                    "放松、保持静止、注视中央",
                    progress,
                    args.prepare_seconds,
                    quality_provider,
                ):
                    break
                symbol, instruction = INSTRUCTIONS[label]
                if not display.show(
                    "任务提示",
                    symbol,
                    "看清方向，保持放松\n等待“开始运动想象”后再想象",
                    progress,
                    args.cue_seconds,
                    quality_provider,
                ):
                    break

                start_sample, trial_segment = recorder.cursor()
                cue_time = time.monotonic()
                active_trial = {
                    "trial_index": trial_index, "block": block, "label": label,
                    "start_sample": start_sample, "segment_id": trial_segment,
                    "cue_monotonic": cue_time,
                }
                if not display.show(
                    "开始运动想象",
                    symbol,
                    instruction,
                    progress,
                    args.imagery_seconds,
                    quality_provider,
                ):
                    discarded_trials.append({**active_trial, "reason": "user_interrupted"})
                    active_trial = None
                    break
                stop_sample = start_sample + int(
                    round(args.imagery_seconds * actual_sample_rate)
                )
                ready = recorder.wait_for_samples(stop_sample, timeout=2.0, segment_id=trial_segment)
                if not ready:
                    reason = "stream_interrupted" if recorder.recovering or recorder.segment_id != trial_segment else "insufficient_samples"
                    discarded_trials.append({**active_trial, "reason": reason, "stop_sample": recorder.sample_count})
                    print(f"Trial {trial_index + 1} 已丢弃：{reason}；保留此前完成的 trial。")
                else:
                    try:
                        epoch = recorder.segment(start_sample, stop_sample, segment_id=trial_segment)
                    except ValueError as exception:
                        discarded_trials.append({**active_trial, "reason": "segment_boundary", "detail": str(exception)})
                    else:
                        trial = TrialRecord(
                            trial_index=trial_index,
                            block=block,
                            label=label,
                            start_sample=start_sample,
                            stop_sample=stop_sample,
                            cue_monotonic=cue_time,
                            artifact=display.consume_artifact(),
                            segment_id=trial_segment,
                        )
                        trials.append(trial)
                        epochs.append(epoch)
                        labels.append(label)
                active_trial = None

                rest = rng.uniform(args.rest_min, args.rest_max)
                progress = (
                    f"第 {block + 1}/{args.blocks} 组   第 {trial_index + 1}/{total_trials} 次\n"
                    + trial_progress(trials, discarded_trials)
                )
                if not display.show(
                    "休息",
                    "+",
                    "停止想象，眨眼并放松",
                    progress,
                    rest,
                    quality_provider,
                ):
                    break
    except BaseException as exception:
        recording_error = exception
        recording_traceback = exception.__traceback__
        if active_trial is not None:
            discarded_trials.append({**active_trial, "reason": "capture_error", "detail": str(exception)})
    finally:
        continuous, timestamps = recorder.snapshot()
        display.close()

    if not epochs and not discarded_trials:
        if recording_error is not None:
            raise recording_error.with_traceback(recording_traceback)
        print("没有完成的 trial，未写入记录文件。")
        return 2
    destination = save_mi_recording(
        output,
        continuous=continuous,
        timestamps=timestamps,
        epochs=np.stack(epochs) if epochs else np.empty((0, len(actual_channel_names), int(round(args.imagery_seconds * actual_sample_rate)))),
        labels=labels,
        trials=trials,
        sample_rate=actual_sample_rate,
        channel_names=actual_channel_names,
        subject_id=args.subject,
        session_id=args.session,
        segment_ids=recorder.snapshot_segment_ids(),
        discarded_trials=discarded_trials,
        recovery_stats=dict(getattr(source, "recovery_stats", {})),
        recovery_events=list(getattr(source, "recovery_events", [])),
        configuration={
            **configuration,
            "sample_rate": actual_sample_rate,
        },
    )
    print(
        f"已保存 {len(epochs)} 个 trial 到 {destination}；"
        f"有效（未标记伪迹）{sum(not trial.artifact for trial in trials)} 个，"
        f"伪迹标记 {sum(trial.artifact for trial in trials)} 个，断流或中断丢弃 {len(discarded_trials)} 个。"
    )
    if recording_error is not None:
        print("采集发生异常；以上已完成 trial 已作为部分记录安全保存。")
        raise recording_error.with_traceback(recording_traceback)
    return 3 if discarded_trials else 0


if __name__ == "__main__":
    raise SystemExit(main())

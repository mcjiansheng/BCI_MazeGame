"""Run a visual motor-imagery calibration experiment and save labeled EEG."""

from __future__ import annotations

import argparse
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


INSTRUCTIONS = {
    "left_hand": ("←", "想象左手反复握紧和放松\n保持身体完全不动"),
    "right_hand": ("→", "想象右手反复握紧和放松\n保持身体完全不动"),
    "feet": ("↑", "想象双脚同时踩踏或脚趾运动\n不要真的移动双脚"),
    "tongue": ("●", "想象舌头反复活动\n不要真的移动舌头或下颌"),
}


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
        if fullscreen:
            self.root.attributes("-fullscreen", True)
        self.phase = tk.Label(self.root, text="准备", font=("Microsoft YaHei", 32), fg="#aab4c8", bg="#10141d")
        self.phase.pack(pady=(50, 10))
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
        self.quality = tk.Label(self.root, text="信号检查中…", font=("Microsoft YaHei", 14), fg="#8bd49c", bg="#10141d")
        self.quality.pack(side="bottom", pady=30)
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
                good, total = quality_provider()
                if not self._artifact:
                    color = "#8bd49c" if good == total else "#ffcc66"
                    self.quality.configure(
                        text=f"实时信号：{good}/{total} 通道未发现明显异常  |  操作员按 A 标记伪迹",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--subject", required=True, help="anonymous participant ID, e.g. S01")
    parser.add_argument("--session", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--classes", default="left_hand,right_hand")
    parser.add_argument("--trials-per-class", type=int, default=40)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--prepare-seconds", type=float, default=2.0)
    parser.add_argument("--cue-seconds", type=float, default=1.0)
    parser.add_argument("--imagery-seconds", type=float, default=4.0)
    parser.add_argument("--rest-min", type=float, default=2.0)
    parser.add_argument("--rest-max", type=float, default=3.0)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    classes = tuple(part.strip() for part in args.classes.split(",") if part.strip())
    if (
        len(classes) < 2
        or len(set(classes)) != len(classes)
        or any(label not in INSTRUCTIONS for label in classes)
    ):
        raise ValueError(
            f"Choose at least two distinct classes from {tuple(INSTRUCTIONS)}"
        )
    schedule = make_schedule(classes, args.blocks, args.trials_per_class, args.seed)
    output = args.output or Path("data/recordings") / args.subject / f"{args.session}.npz"
    source = source_from_args(args)
    display = CueDisplay(args.fullscreen)
    trials: list[TrialRecord] = []
    epochs: list[np.ndarray] = []
    labels: list[str] = []
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

            def quality_provider() -> tuple[int, int]:
                if recorder.error is not None:
                    raise RuntimeError("EEG recorder stopped unexpectedly") from recorder.error
                needed = actual_sample_rate * 2
                data = recorder.recent(needed)
                if data.shape[1] < needed:
                    return 0, len(actual_channel_names)
                quality = assess_signal_quality(data, actual_sample_rate)
                return sum(item.status != "bad" for item in quality), len(quality)

            if not display.show(
                "设备预热",
                "+",
                "坐直、放松肩颈、注视中央\n不要说话或移动",
                "预热与信号检查",
                args.warmup_seconds,
                quality_provider,
            ):
                return 1

            total_trials = len(schedule)
            for trial_index, (block, label) in enumerate(schedule):
                display.begin_trial()
                progress = f"Block {block + 1}/{args.blocks}   Trial {trial_index + 1}/{total_trials}"
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
                    instruction,
                    progress,
                    args.cue_seconds,
                    quality_provider,
                ):
                    break

                start_sample = recorder.sample_count
                cue_time = time.monotonic()
                if not display.show(
                    "开始运动想象",
                    symbol,
                    instruction,
                    progress,
                    args.imagery_seconds,
                    quality_provider,
                ):
                    break
                stop_sample = start_sample + int(
                    round(args.imagery_seconds * actual_sample_rate)
                )
                if not recorder.wait_for_samples(stop_sample, timeout=2.0):
                    raise RuntimeError(
                        f"EEG stream did not deliver {stop_sample - start_sample} samples in time"
                    )
                epoch = recorder.segment(start_sample, stop_sample)
                trial = TrialRecord(
                    trial_index=trial_index,
                    block=block,
                    label=label,
                    start_sample=start_sample,
                    stop_sample=stop_sample,
                    cue_monotonic=cue_time,
                    artifact=display.consume_artifact(),
                )
                trials.append(trial)
                epochs.append(epoch)
                labels.append(label)

                rest = rng.uniform(args.rest_min, args.rest_max)
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
    finally:
        continuous, timestamps = recorder.snapshot()
        display.close()

    if not epochs:
        if recording_error is not None:
            raise recording_error.with_traceback(recording_traceback)
        print("没有完成的 trial，未写入记录文件。")
        return 2
    destination = save_mi_recording(
        output,
        continuous=continuous,
        timestamps=timestamps,
        epochs=np.stack(epochs),
        labels=labels,
        trials=trials,
        sample_rate=actual_sample_rate,
        channel_names=actual_channel_names,
        subject_id=args.subject,
        session_id=args.session,
        configuration={
            "backend": args.backend,
            "device_channels_1_based": [value + 1 for value in args.channels],
            "prepare_seconds": args.prepare_seconds,
            "cue_seconds": args.cue_seconds,
            "imagery_seconds": args.imagery_seconds,
            "rest_range_seconds": [args.rest_min, args.rest_max],
            "seed": args.seed,
        },
    )
    print(
        f"已保存 {len(epochs)} 个 trial 到 {destination}；"
        f"伪迹标记 {sum(trial.artifact for trial in trials)} 个。"
    )
    if recording_error is not None:
        print("采集发生异常；以上已完成 trial 已作为部分记录安全保存。")
        raise recording_error.with_traceback(recording_traceback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

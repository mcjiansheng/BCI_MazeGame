"""Paired rest/task acquisition for movement and motor-imagery experiments."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np

from bci_maze.lk_mini.cli import source_from_args
from bci_maze.lk_mini.quality import assess_signal_quality
from bci_maze.lk_mini.recording import (
    ContinuousRecorder, TrialRecord, save_mi_recording as _save_standard_recording,
)
from bci_maze.lk_mini.source import DEFAULT_CHANNEL_NAMES


TASKS = {"motor": "motor_execution", "imagery": "motor_imagery", "benchmark": "motor_imagery"}
LABELS = ("left_hand", "right_hand")
BASELINE_SECONDS = 2.0
CUE_SECONDS = 1.0
ACTIVE_SECONDS = 4.0
BASELINE_PROTOCOL = {
    "kind": "within_trial_pre_cue_rest", "duration_seconds": BASELINE_SECONDS,
    "sequence": ["baseline", "cue", "active", "rest"],
    "instruction": "放松双手、保持静止、注视中央；方向提示在基线结束后出现。",
    "interval_convention": "start_sample inclusive, stop_sample exclusive",
}


@dataclass(frozen=True)
class CollectionResult:
    path: Path | None
    completed: int
    discarded: int
    interrupted: bool = False
    error: str = ""
    summary: dict | None = None


def make_schedule(trials_per_class: int, blocks: int, seed: int) -> list[tuple[int, str]]:
    if (type(trials_per_class) is not int or type(blocks) is not int
            or blocks <= 0 or trials_per_class < blocks or trials_per_class % blocks):
        raise ValueError("每类试次数须为组数的正整数倍。")
    rng = random.Random(seed)
    schedule = []
    for block in range(blocks):
        labels = list(LABELS) * (trials_per_class // blocks)
        rng.shuffle(labels)
        schedule.extend((block, label) for label in labels)
    return schedule


def emit_progress(phase: str, *, label: str = "", trial_index: int = 0,
                  total_trials: int = 0, block: int = 0, duration: float = 0,
                  text: str = "", **metrics) -> None:
    print("EXPERIMENT_EVENT " + json.dumps(dict(
        phase=phase, label=label, trial_index=trial_index, total_trials=total_trials,
        block=block, duration=duration, started_at=time.time(), text=text, **metrics,
    ), ensure_ascii=False), flush=True)


def save_mi_recording(path, *, baseline_epochs, baseline_intervals, **kwargs) -> Path:
    """Atomically save aligned baseline/task arrays inside the stop-protected frame.

    The dashboard worker defers stop signals while a function named
    ``save_mi_recording`` remains on the main-thread stack, covering both writes
    and the final rename without changing the original recording implementation.
    """
    path = Path(path)
    epochs = np.asarray(kwargs["epochs"])
    baseline = np.asarray(baseline_epochs)
    n = len(kwargs["trials"])
    if (epochs.ndim != 3 or baseline.ndim != 3 or epochs.shape[:2] != baseline.shape[:2]
            or epochs.shape[0] != n or len(kwargs["labels"]) != n or len(baseline_intervals) != n):
        raise ValueError("基线、任务片段、标签和试次信息须逐项对应。")
    rate = int(kwargs["sample_rate"])
    if baseline.shape[2] != int(round(rate * BASELINE_SECONDS)):
        raise ValueError("每个基线片段须包含完整的 2 秒样本。")
    if epochs.shape[2] != int(round(rate * ACTIVE_SECONDS)):
        raise ValueError("每个任务片段须包含完整的 4 秒样本。")
    continuous = np.asarray(kwargs["continuous"])
    ids = np.asarray(kwargs.get("segment_ids", np.zeros(continuous.shape[1], dtype=np.int64)))
    if continuous.ndim != 2 or ids.shape != (continuous.shape[1],):
        raise ValueError("连续信号与连接段标记须对应。")
    for index, (trial, interval) in enumerate(zip(kwargs["trials"], baseline_intervals)):
        start, stop, segment = (int(interval[key]) for key in ("start_sample", "stop_sample", "segment_id"))
        if (start < 0 or stop - start != baseline.shape[2] or stop > trial.start_sample
                or trial.stop_sample > continuous.shape[1] or trial.stop_sample - trial.start_sample != epochs.shape[2]
                or trial.segment_id != segment or not np.all(ids[start:trial.stop_sample] == segment)):
            raise ValueError("基线、提示及任务区间必须完整且位于同一连接段。")
        if (not np.array_equal(baseline[index], continuous[:, start:stop], equal_nan=True)
                or not np.array_equal(epochs[index], continuous[:, trial.start_sample:trial.stop_sample], equal_nan=True)):
            raise ValueError("基线及任务数组须与保存的连续信号区间一致。")
    configuration = dict(kwargs["configuration"])
    configuration.update(baseline_protocol=dict(BASELINE_PROTOCOL),
                         baseline_intervals=[dict(item) for item in baseline_intervals],
                         baseline_seconds=BASELINE_SECONDS, cue_seconds=CUE_SECONDS,
                         active_seconds=ACTIVE_SECONDS)
    kwargs = dict(kwargs, configuration=configuration)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        _save_standard_recording(temporary, **kwargs)
        with np.load(temporary, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        summary = configuration.get("collection_summary")
        if summary:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["trial_summary"].update(
                planned=summary["target_trials"], completed=summary["complete_trials"],
                usable=summary["usable_trials"], artifact_marked=summary["artifact_trials"],
                nonfinite=summary["nonfinite_trials"], excluded_complete=summary["excluded_complete_trials"],
                discarded=summary["incomplete_trials"], unattempted=summary["unattempted_trials"],
                attempts=summary["total_attempts"], retries=summary["retry_count"], unfilled=summary["unfilled_trials"],
                planned_definition="planned slots; each slot targets one usable trial",
                completed_definition="complete recorded attempts including artifact-marked and nonfinite trials",
                usable_definition="complete attempts passing operator-artifact and finite-value checks",
                unattempted_definition="planned slots with no attempt started",
                unfilled_definition="planned slots without a usable completed attempt",
            )
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        arrays.update(
            baseline_epochs=baseline.astype(np.float32),
            baseline_start_samples=np.asarray([item["start_sample"] for item in baseline_intervals], dtype=np.int64),
            baseline_stop_samples=np.asarray([item["stop_sample"] for item in baseline_intervals], dtype=np.int64),
            baseline_segment_ids=np.asarray([item["segment_id"] for item in baseline_intervals], dtype=np.int64),
        )
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def extract_trial_pair(recorder, baseline_cursor: tuple[int, int],
                       active_cursor: tuple[int, int], sample_rate: int):
    """Reject a gap anywhere from the baseline through the end of the task."""
    baseline_start, segment = baseline_cursor
    active_start, active_segment = active_cursor
    baseline_stop = baseline_start + int(round(BASELINE_SECONDS * sample_rate))
    active_stop = active_start + int(round(ACTIVE_SECONDS * sample_rate))
    if active_segment != segment:
        raise ValueError("stream_segment_changed")
    if baseline_stop > active_start or recorder.sample_count < active_stop:
        raise ValueError("incomplete_baseline_or_active_window")
    combined = recorder.segment(baseline_start, active_stop, segment_id=segment)
    baseline = combined[:, :baseline_stop - baseline_start].copy()
    epoch = combined[:, active_start - baseline_start:].copy()
    if baseline.shape[1] != baseline_stop - baseline_start or epoch.shape[1] != active_stop - active_start:
        raise ValueError("incomplete_baseline_or_active_window")
    return baseline, epoch, dict(start_sample=baseline_start, stop_sample=baseline_stop, segment_id=segment)


def _display_factory(fullscreen: bool):
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.collect_lk_mi import CueDisplay
    return CueDisplay(fullscreen)


def collect_experiment(args, *, source_factory=None, recorder_factory=None,
                       display_factory=None) -> CollectionResult:
    if args.experiment not in {"motor", "imagery"}:
        raise ValueError("实时采集支持 motor 或 imagery；benchmark 使用已有运动想象记录。")
    if args.backend == "replay":
        raise ValueError("已有记录请使用 analyze；新提示采集请选择设备、LSL 或合成源。")
    if getattr(args, "no_realtime", False):
        raise ValueError("提示采集需要实时采样节奏；离线合成示例请使用 demo。")
    schedule = make_schedule(args.trials_per_class, args.blocks, args.seed)
    max_attempts = int(getattr(args, "max_attempts_per_slot", 3))
    wait_limit = float(getattr(args, "stable_wait_timeout", 60.0))
    duration_limit = float(getattr(args, "max_duration_seconds", 0) or max(300, len(schedule) * 30))
    if not 1 <= max_attempts <= 3 or not np.isfinite([wait_limit, duration_limit]).all() or min(wait_limit, duration_limit) <= 0:
        raise ValueError("每个计划试次最多尝试 1–3 次，等待及整场时限须为正有限秒数。")
    source = (source_factory or source_from_args)(args)
    recorder = (recorder_factory or ContinuousRecorder)(source)
    display = (display_factory or _display_factory)(getattr(args, "fullscreen", False))
    task = TASKS[args.experiment]
    task_name = "实际左右手动作" if task == "motor_execution" else "左右手运动想象"
    display.set_plan(f"{task_name} · 静息 2 秒 → 提示 1 秒 → 任务 4 秒 → 休息 2–3 秒 · 目标 {len(schedule)} 次有效记录")
    if hasattr(display, "root"):
        display.root.title(f"MindMaze 子实验 · {task_name}")
    trials, epochs, baselines, intervals, discarded, attempts, waits = [], [], [], [], [], [], []
    rng = random.Random(args.seed + 1)
    pending = None
    interrupted, display_closed = False, False
    error = stop_reason = ""
    retained, slot_index, retry_index, block, label = 0, 0, 0, 0, ""
    started = time.monotonic()
    rate, names = int(source.sample_rate), tuple(source.channel_names)

    def quality_provider():
        if recorder.error is not None:
            raise RuntimeError("采集器停止：" + str(recorder.error)) from recorder.error
        if recorder.recovering:
            return None, len(names), "连接恢复中；等待新的完整信号窗口"
        if not fresh_stream():
            return None, len(names), "等待最近 0.5 秒内的新样本，暂不可评估"
        data = recorder.recent(rate * 2)
        if data.shape[1] < rate * 2:
            return None, len(names), "等待当前连接段的完整 2 秒信号"
        quality = assess_signal_quality(data, rate)
        details = "  ".join(f"{name}: {q.status}" for name, q in zip(names, quality) if q.status != "good")
        return sum(q.status == "good" for q in quality), len(names), details or "全部通道良好"

    def fresh_stream():
        age = getattr(recorder, "last_sample_age", 0.0)
        return age is not None and age <= .5

    def counters():
        return dict(attempt_index=len(attempts) - 1, total_attempts=len(attempts),
                    retained_trials=retained, target_trials=len(schedule),
                    discarded_trials=sum(item.get("status") in {"discarded", "excluded"} for item in attempts),
                    retry_count=retry_index)

    def check_deadline(duration=0):
        if time.monotonic() - started + duration > duration_limit:
            raise TimeoutError("session_duration_limit")

    def show(phase, title, symbol, instruction, duration):
        check_deadline(duration)
        if recorder.error is not None:
            raise RuntimeError("采集器停止：" + str(recorder.error)) from recorder.error
        phase_cursor = recorder.cursor()
        marker = dict(phase=phase, requested_at=time.time(), requested_monotonic=time.monotonic(),
                      duration_seconds=duration, start_sample=phase_cursor[0],
                      start_segment_id=phase_cursor[1], last_sample_age=getattr(recorder, "last_sample_age", None))
        if pending is not None:
            pending["phase_timings"].append(marker)
        emit_progress(phase, label=label, trial_index=slot_index, total_trials=len(schedule),
                      block=block, duration=duration, text=instruction, **counters())
        try:
            result = display.show(title, symbol, instruction,
                                  f"第 {block + 1}/{args.blocks} 组 · 目标 {slot_index + 1}/{len(schedule)} · "
                                  f"本次尝试 {retry_index + 1}/{max_attempts} · 有效 {retained}/{len(schedule)}",
                                  duration, quality_provider)
            marker["completed"] = bool(result)
            return result
        finally:
            marker.update(ended_at=time.time(), ended_monotonic=time.monotonic(),
                          stop_sample=recorder.cursor()[0], stop_segment_id=recorder.cursor()[1])

    def wait_until_stable():
        waiting = dict(planned_slot=slot_index, retry_index=retry_index, started_at=time.time(),
                       started_monotonic=time.monotonic())
        waits.append(waiting)
        try:
            while recorder.recovering or recorder.recent(rate * 2).shape[1] < rate * 2 or not fresh_stream():
                if time.monotonic() - waiting["started_monotonic"] >= wait_limit:
                    raise TimeoutError("stable_wait_timeout")
                instruction = ("连接恢复中，恢复后重新积累完整基线。" if recorder.recovering else
                               "等待当前连接段积累 2 秒信号。" if recorder.recent(rate * 2).shape[1] < rate * 2 else
                               "等待最近 0.5 秒内的新样本，再开始任务。")
                if not show("waiting", "等待稳定数据", "+", instruction, .5):
                    return False
            return True
        finally:
            waiting.update(ended_at=time.time(), ended_monotonic=time.monotonic(),
                           sample_count=recorder.sample_count, segment_id=recorder.segment_id)

    def wait_window(start, stop, segment):
        # Current recovery state does not invalidate an already complete interval.
        if recorder.sample_count < stop:
            recorder.wait_for_samples(stop, timeout=min(2.0, max(0.0, duration_limit - (time.monotonic() - started))),
                                      segment_id=segment)
        if recorder.sample_count < stop:
            raise ValueError("incomplete_window")
        recorder.segment(start, stop, segment_id=segment)

    def begin_attempt(attempt_id, baseline_cursor):
        def finish():
            nonlocal pending
            pending = dict(trial_index=attempt_id, attempt_index=attempt_id, planned_slot=slot_index,
                           retry_index=retry_index, block=block, label=label, status="running",
                           started_at=time.time(), started_monotonic=time.monotonic(), phase_timings=[],
                           baseline_start_sample=baseline_cursor[0], segment_id=baseline_cursor[1],
                           baseline_stop_sample=baseline_cursor[0] + round(rate * BASELINE_SECONDS))
            attempts.append(pending)
        return finish()

    def end_attempt(status, reason="", detail=""):
        def finish():
            # Keep state transition, rejection bookkeeping and pending cleanup
            # indivisible with respect to the dashboard's cooperative stop.
            nonlocal pending
            pending.update(status=status, reason=reason, detail=detail, ended_at=time.time(),
                           ended_monotonic=time.monotonic(), end_sample_count=recorder.sample_count,
                           end_segment_id=recorder.segment_id, recovering_at_end=bool(recorder.recovering))
            for phase in ("baseline", "active"):
                target = pending.get(phase + "_stop_sample")
                if target is not None:
                    pending[phase + "_missing_samples"] = max(0, target - recorder.sample_count)
            if status == "discarded":
                discarded.append(dict(pending))
            pending = None
        return finish()

    def commit_trial(attempt_id, active_cursor, baseline, epoch, interval):
        def finish():
            """Commit aligned arrays, usable count and attempt status together."""
            nonlocal retained
            artifact = bool(display.consume_artifact())
            finite = bool(np.isfinite(epoch).all() and np.isfinite(baseline).all())
            trials.append(TrialRecord(attempt_id, block, label, active_cursor[0], pending["active_stop_sample"],
                                      pending["cue_monotonic"], artifact, active_cursor[1]))
            epochs.append(epoch)
            baselines.append(baseline)
            intervals.append(dict(interval, trial_index=attempt_id, planned_slot=slot_index))
            accepted = not artifact and finite
            if accepted:
                retained += 1
            pending.update(artifact=artifact, finite=finite, stored_trial_index=len(trials) - 1)
            end_attempt("retained" if accepted else "excluded",
                        "" if accepted else "manual_artifact_and_nonfinite" if artifact and not finite
                        else "manual_artifact" if artifact else "nonfinite")
            return accepted
        return finish()

    def finish():
        """Protect cleanup, all snapshots, array assembly and atomic persistence."""
        nonlocal pending, display_closed
        if not display_closed:
            display.close()
            display_closed = True
        if pending is not None:
            end_attempt("discarded", "user_interrupted" if interrupted else stop_reason or "capture_error", error)
        continuous, timestamps = recorder.snapshot()
        manual = [bool(trial.artifact) for trial in trials]
        finite = [bool(np.isfinite(epoch).all() and np.isfinite(base).all()) for epoch, base in zip(epochs, baselines)]
        invalid = sum(mark or not valid for mark, valid in zip(manual, finite))
        summary = dict(target_trials=len(schedule), total_attempts=len(attempts), complete_trials=len(trials),
                       usable_trials=len(trials) - invalid, artifact_trials=sum(manual),
                       nonfinite_trials=sum(not valid for valid in finite), excluded_complete_trials=invalid,
                       incomplete_trials=len(discarded), retry_count=sum(item["retry_index"] > 0 for item in attempts),
                       target_met=retained == len(schedule), elapsed_seconds=time.monotonic() - started,
                       unattempted_trials=len(schedule) - len({item["planned_slot"] for item in attempts}),
                       unfilled_trials=len(schedule) - retained,
                       status="interrupted" if interrupted else "time_limit" if stop_reason in {"stable_wait_timeout", "session_duration_limit"}
                       else "error" if error else "target_met" if retained == len(schedule) else "target_not_met",
                       stop_reason="user_interrupted" if interrupted else stop_reason or ("target_met" if retained == len(schedule) else "attempt_limits_reached"),
                       max_attempts_per_slot=max_attempts, stable_wait_timeout_seconds=wait_limit,
                       max_duration_seconds=duration_limit, freshness_limit_seconds=.5, last_planned_slot=slot_index,
                       last_retry_index=retry_index,
                       raw_samples=int(continuous.shape[1]), raw_seconds=continuous.shape[1] / rate)
        path = save_mi_recording(
            args.recording_output, continuous=continuous, timestamps=timestamps,
            epochs=np.stack(epochs) if epochs else np.empty((0, len(names), round(rate * ACTIVE_SECONDS))),
            baseline_epochs=np.stack(baselines) if baselines else np.empty((0, len(names), round(rate * BASELINE_SECONDS))),
            baseline_intervals=intervals, labels=[trial.label for trial in trials], trials=trials, sample_rate=rate,
            channel_names=names, subject_id=args.subject, session_id=args.session,
            segment_ids=recorder.snapshot_segment_ids(), discarded_trials=discarded,
            recovery_stats=dict(source.recovery_stats), recovery_events=list(source.recovery_events),
            configuration=dict(task=task, backend=args.backend, label_source="visual_cue",
                               planned_trials=len(schedule), blocks=args.blocks, seed=args.seed,
                               capture_interrupted=interrupted, capture_error=error,
                               wear_state=getattr(args, "wear_state", "unknown"),
                               attempt_history=attempts, stable_wait_history=waits, collection_summary=summary,
                               planned_schedule=[dict(planned_slot=i, block=b, label=l) for i, (b, l) in enumerate(schedule)],
                               reference="SRB2" if args.backend == "brainflow" else "unspecified",
                               bias="BIAS" if args.backend == "brainflow" else "unspecified",
                               gain=getattr(args, "gain", 24) if args.backend == "brainflow" else None,
                               timing_basis="continuous_recorder_sample_indices_at_phase_request",
                               timestamp_basis={"brainflow": "nominal_rate_reconstruction_per_connection_segment",
                                                "lsl": "lsl_stream_timestamps", "synthetic": "synthetic_nominal_sample_clock"}[args.backend]),
        )
        emit_progress("complete", trial_index=slot_index, total_trials=len(schedule), block=block,
                      text=f"已保存 {retained}/{len(schedule)} 次有效记录及全部已收到的连续信号。", **counters())
        print(f"实验记录：{path}", flush=True)
        return CollectionResult(path, len(trials), len(discarded), interrupted, error, summary)

    try:
        try:
            with recorder:
                rate, names = int(source.sample_rate), tuple(source.channel_names)
                if rate <= 0 or not names:
                    raise ValueError("数据源须提供有效采样率与通道名称。")
                if not show("waiting", "设备预热", "+", "放松双手、保持静止，等待设备收数。", getattr(args, "warmup_seconds", 5.0)):
                    interrupted = True
                for slot_index, (block, label) in enumerate(schedule):
                    if interrupted or display.aborted:
                        break
                    for retry_index in range(max_attempts):
                        if not wait_until_stable():
                            interrupted = True
                            break
                        check_deadline()
                        display.begin_trial()
                        baseline_cursor = recorder.cursor()
                        attempt_id = len(attempts)
                        begin_attempt(attempt_id, baseline_cursor)
                        accepted = False
                        try:
                            if not show("baseline", "静息基线", "+", "放松双手、保持静止、注视中央。", BASELINE_SECONDS):
                                interrupted = True
                                break
                            wait_window(baseline_cursor[0], pending["baseline_stop_sample"], baseline_cursor[1])
                            symbol = "←" if label == "left_hand" else "→"
                            side = "左手" if label == "left_hand" else "右手"
                            if not show("cue", "方向提示", symbol, f"准备{side}任务，保持放松，等待开始。", CUE_SECONDS):
                                interrupted = True
                                break
                            if not fresh_stream() or recorder.recovering:
                                raise ValueError("stale_stream_before_active")
                            active_cursor = recorder.cursor()
                            if active_cursor[1] != baseline_cursor[1]:
                                raise ValueError("stream_segment_changed_before_active")
                            pending.update(start_sample=active_cursor[0], active_start_sample=active_cursor[0],
                                           active_segment_id=active_cursor[1], active_stop_sample=active_cursor[0] + round(rate * ACTIVE_SECONDS),
                                           cue_monotonic=time.monotonic())
                            instruction = (f"{side}以舒适力度握紧、放松手指，约每秒一次。\n其余身体部位保持放松。"
                                           if task == "motor_execution" else f"保持手和身体不动，想象{side}反复握紧和放松。")
                            if not show("active", "实际握拳" if task == "motor_execution" else "运动想象", symbol, instruction, ACTIVE_SECONDS):
                                interrupted = True
                                break
                            wait_window(active_cursor[0], pending["active_stop_sample"], active_cursor[1])
                            baseline, epoch, interval = extract_trial_pair(recorder, baseline_cursor, active_cursor, rate)
                        except ValueError as exception:
                            end_attempt("discarded", str(exception))
                            print(f"计划第 {slot_index + 1} 次，尝试 {retry_index + 1} 未完成：{exception}", flush=True)
                        else:
                            accepted = commit_trial(attempt_id, active_cursor, baseline, epoch, interval)
                        if not show("rest", "休息", "+", "停止任务，放松双手，可自然眨眼。", rng.uniform(2, 3)):
                            interrupted = True
                            break
                        if accepted:
                            break
                    if interrupted:
                        break
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            stop_reason = str(exception) if isinstance(exception, TimeoutError) else "capture_error"
        return finish()
    except KeyboardInterrupt:
        # One retry handles a late stop at finalization entry; further direct
        # interrupts still propagate. Dashboard stop is delivered only once.
        interrupted = True
        return finish()


def create_demo_recording(path, *, experiment: str, subject: str, session: str,
                          trials_per_class: int = 10, blocks: int = 2, seed: int = 42,
                          sample_rate: int = 250, channel_names=DEFAULT_CHANNEL_NAMES) -> Path:
    """Generate reproducible, explicitly synthetic contralateral mu/beta changes."""
    if experiment not in TASKS:
        raise ValueError("未知子实验。")
    if experiment == "benchmark":
        trials_per_class, blocks = 20, 4  # 40 trials, four independent block groups.
    schedule = make_schedule(trials_per_class, blocks, seed)
    names = tuple(channel_names)
    if sample_rate < 100 or "C3" not in names or "C4" not in names:
        raise ValueError("合成示例需要至少 100 Hz 采样率和 C3、C4 通道。")
    rng = np.random.default_rng(seed)
    parts, trials, epochs, baselines, intervals = [], [], [], [], []
    position = 0
    for index, (block, label) in enumerate(schedule):
        counts = [round(sample_rate * BASELINE_SECONDS), round(sample_rate * CUE_SECONDS),
                  round(sample_rate * ACTIVE_SECONDS), round(sample_rate * rng.uniform(2, 3))]
        count = sum(counts)
        t = np.arange(count) / sample_rate
        phases = rng.uniform(0, 2 * np.pi, (len(names), 2))
        mu = rng.uniform(16, 20, (len(names), 1)) * np.sin(2 * np.pi * 10 * t + phases[:, 0:1])
        beta = rng.uniform(8, 11, (len(names), 1)) * np.sin(2 * np.pi * 20 * t + phases[:, 1:2])
        active_local = counts[0] + counts[1]
        active_slice = slice(active_local, active_local + counts[2])
        contralateral = names.index("C4" if label == "left_hand" else "C3")
        ipsilateral = names.index("C3" if label == "left_hand" else "C4")
        mu[contralateral, active_slice] *= .45
        beta[contralateral, active_slice] *= .55
        mu[ipsilateral, active_slice] *= .95
        beta[ipsilateral, active_slice] *= .95
        data = mu + beta + rng.normal(0, 2, (len(names), count))
        parts.append(data)
        baselines.append(data[:, :counts[0]].copy())
        epochs.append(data[:, active_slice].copy())
        intervals.append(dict(trial_index=index, start_sample=position,
                              stop_sample=position + counts[0], segment_id=0))
        start = position + active_local
        trials.append(TrialRecord(index, block, label, start, start + counts[2], start / sample_rate))
        position += count
    continuous = np.concatenate(parts, axis=1)
    return save_mi_recording(
        path, continuous=continuous, timestamps=np.arange(position) / sample_rate,
        epochs=np.stack(epochs), baseline_epochs=np.stack(baselines), baseline_intervals=intervals,
        labels=[trial.label for trial in trials], trials=trials, sample_rate=sample_rate,
        channel_names=names, subject_id=subject, session_id=session,
        segment_ids=np.zeros(position, dtype=np.int64),
        configuration=dict(task=TASKS[experiment], backend="synthetic_mi",
                           label_source="scripted_simulation", human_mi_verified=False,
                           planned_trials=len(trials), blocks=blocks, seed=seed,
                           timing_basis="synthetic_sample_clock",
                           simulation="10 Hz mu and 20 Hz beta; contralateral C3/C4 amplitude suppression with independent noise"),
    )

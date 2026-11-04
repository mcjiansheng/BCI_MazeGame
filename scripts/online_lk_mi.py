"""Classify live LK-Mini motor imagery with smoothing and low-confidence rejection."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bci_maze.lk_mini.cli import add_source_arguments, source_from_args
from bci_maze.lk_mini.model import load_model_bundle
from bci_maze.lk_mini.display import configure_plot_fonts
from bci_maze.lk_mini.timing import StreamTimingGuard
from bci_maze.lk_mini.transport_log import TransportLogger
from bci_maze.lk_mini.game_control import GameControlWorker, model_control_provenance
from bci_maze.lk_mini.online import (
    ProbabilitySmoother,
    RollingEEGBuffer,
    classify_mi_window,
    validate_stream_metadata,
)


def load_classifier(path: Path, device: str = "auto"):
    if path.suffix.lower() == ".pt":
        from bci_maze.lk_mini.semhaf import load_semhaf_bundle
        return load_semhaf_bundle(path, device=device)
    return load_model_bundle(path)


@contextmanager
def game_lifetime(game):
    """Invalidate pending decisions before the enclosing EEG source begins cleanup."""
    try:
        yield
    finally:
        if game is not None:
            game.set_stream_ready(False)
            game.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--transport-log", type=Path, help="optional independent JSONL of stream reads, state changes and window resets")
    parser.add_argument("--unity-host")
    parser.add_argument("--unity-port", type=int, default=7777)
    parser.add_argument("--control-mode", choices=("junction", "continuous"), default="junction")
    parser.add_argument("--allow-unvalidated-control", action="store_true", help="explicitly allow public/synthetic or unverified models to control Unity for software validation")
    parser.add_argument("--command-cooldown", type=float, default=1.5)
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=5.0,
        help="fail after this many seconds without samples (ignored after replay EOF)",
    )
    args = parser.parse_args()
    if args.step_seconds <= 0 or args.stream_timeout <= 0:
        raise ValueError("--step-seconds and --stream-timeout must be positive")

    bundle = load_classifier(args.model, args.device)
    provenance = dict(getattr(bundle, "provenance", {}) or {})
    if args.unity_host:
        provenance = model_control_provenance(bundle, allow_unvalidated=args.allow_unvalidated_control)
    source = source_from_args(args)
    window_samples = int(round(bundle.config.epoch_seconds * bundle.config.sample_rate))
    step_samples = max(1, int(round(args.step_seconds * bundle.config.sample_rate)))
    buffer = RollingEEGBuffer(len(bundle.channel_names), window_samples + step_samples)
    buffer_timestamps = np.empty(0, dtype=np.float64)
    window_reset_id = 0
    smoother = ProbabilitySmoother(
        bundle.classes,
        history=args.history,
        confidence_threshold=args.confidence,
        margin_threshold=args.margin,
    )
    decisions_path = args.log or Path("outputs/online") / f"{bundle.subject_id}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)

    figure = axes = bars = status_text = None
    if args.plot:
        configure_plot_fonts()
        plt.ion()
        figure, axes = plt.subplots(figsize=(8, 5))
        figure.subplots_adjust(bottom=0.30)
        bars = axes.bar(bundle.classes, np.zeros(len(bundle.classes)), color="#4c9be8")
        axes.set_ylim(0.0, 1.0)
        axes.set_ylabel("平滑分类概率")
        axes.set_title(f"LK-Mini 在线分类 | {len(bundle.channel_names)} 通道 | {bundle.subject_id}")
        status_text = figure.text(
            0.5, 0.04, f"等待 {bundle.config.epoch_seconds:g} 秒数据窗",
            ha="center", va="bottom", fontsize=11,
        )

    samples_since_decision = 0
    last_command_time = -1e9
    game = GameControlWorker(args.unity_host, args.unity_port, mode=args.control_mode) if args.unity_host else None
    game_window_id = -1
    transport = TransportLogger(args.transport_log)
    capture_end = None
    capture_error = ""
    try:
        with source, game_lifetime(game), decisions_path.open("a", encoding="utf-8") as log_file:
            sample_rate, channel_names = validate_stream_metadata(
                source.sample_rate,
                source.channel_names,
                expected_sample_rate=bundle.config.sample_rate,
                expected_channel_names=bundle.channel_names,
            )
            print(
                f"在线分类已启动：subject={bundle.subject_id}, classes={bundle.classes}, "
                f"window={bundle.config.epoch_seconds}s, step={args.step_seconds}s"
            )
            print(f"模型来源：{provenance.get('data_origin', 'unspecified')}；设备：{getattr(bundle, 'device', args.device)}")
            if provenance:
                print(f"个人校准通过：{provenance.get('calibration_passed', False)}；真人 MI 独立确认：{provenance.get('human_mi_verified', False)}")
            if game is not None:
                game.start()
            started = time.monotonic()
            transport.start(started, sample_rate, channel_names, window_samples)
            deadline = started + args.duration if args.duration > 0 else float("inf")
            timing = StreamTimingGuard(sample_rate)
            last_sample_time = started
            current_segment = 0
            was_recovering = False
            while time.monotonic() < deadline:
                if figure is not None:
                    if not plt.fignum_exists(figure.number):
                        break
                    figure.canvas.flush_events()
                chunk = source.read(timeout=min(0.25, max(0.0, deadline - time.monotonic())))
                read_monotonic = time.monotonic()
                transport.observe(chunk, source, read_monotonic)
                recovering = bool(getattr(source, "recovering", False))
                if int(chunk.segment_id) != current_segment or (recovering and not was_recovering):
                    if game is not None:
                        game.set_stream_ready(False)
                    current_segment = int(chunk.segment_id)
                    buffer = RollingEEGBuffer(len(bundle.channel_names), window_samples + step_samples)
                    buffer_timestamps = np.empty(0, dtype=np.float64)
                    window_reset_id += 1
                    transport.write("window_reset", read_monotonic, window_reset_id=window_reset_id,
                                    segment_id=current_segment, reason="stream_recovering" if recovering else "segment_changed")
                    smoother.reset()
                    timing = StreamTimingGuard(sample_rate)
                    samples_since_decision = 0
                    last_command_time = -1e9
                    last_sample_time = time.monotonic()
                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "segment_id": current_segment,
                        "label": "unknown", "accepted": False,
                        "reason": "stream_recovering" if recovering else "segment_changed",
                        "recovery_stats": dict(getattr(source, "recovery_stats", {})),
                    }
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    print(f"unknown | {record['reason']} | segment={current_segment}")
                    if bars is not None:
                        for bar in bars:
                            bar.set_height(0.0)
                        status_text.set_text("数据流恢复中；恢复后重新积累完整分类窗口")
                        figure.canvas.draw_idle()
                        figure.canvas.flush_events()
                was_recovering = recovering
                game_snapshot = game.snapshot if game is not None else None
                if game is not None:
                    for event in game.drain_events():
                        log_file.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "unity", "reason": "unity_event", "label": "unknown", **event}, ensure_ascii=False) + "\n")
                    log_file.flush()
                    if game_snapshot.window_id != game_window_id:
                        game_window_id = game_snapshot.window_id
                        buffer = RollingEEGBuffer(len(bundle.channel_names), window_samples + step_samples)
                        buffer_timestamps = np.empty(0, dtype=np.float64)
                        window_reset_id += 1
                        transport.write("window_reset", window_reset_id=window_reset_id,
                                        segment_id=current_segment, reason="game_window_changed")
                        smoother.reset()
                        samples_since_decision = 0
                        last_command_time = -1e9
                        if game_snapshot.phase == "waiting":
                            choices = {label: game_snapshot.command_for(label) for label in bundle.classes}
                            print(f"路口 {game_snapshot.visit_token}：{choices}；重新采集完整运动想象窗口。")
                            if status_text is not None:
                                status_text.set_text(f"路口决策：左手={choices.get('left_hand')}，右手={choices.get('right_hand')}\n请从现在开始运动想象")
                                figure.canvas.draw_idle()
                if not chunk.data.shape[1]:
                    if source.exhausted:
                        break
                    if not recovering and time.monotonic() - last_sample_time >= args.stream_timeout:
                        raise RuntimeError(
                            f"EEG stream produced no samples for {args.stream_timeout:g} seconds"
                        )
                    continue
                last_sample_time = time.monotonic()
                timing.check(chunk.monotonic_timestamps)
                data = chunk.data
                data_timestamps = chunk.monotonic_timestamps
                if game_snapshot is not None and game_snapshot.phase == "waiting":
                    keep = chunk.monotonic_timestamps >= game_snapshot.window_started
                    data = data[:, keep]
                    data_timestamps = data_timestamps[keep]
                buffer.append(data)
                buffer_timestamps = np.concatenate((buffer_timestamps, data_timestamps))[-buffer.capacity_samples:]
                samples_since_decision += data.shape[1]
                if buffer.sample_count < window_samples or samples_since_decision < step_samples:
                    continue
                samples_since_decision %= step_samples
                window = buffer.latest(window_samples)
                window_result = classify_mi_window(
                    bundle, smoother, window, sample_rate, channel_names
                )
                decision = window_result.decision
                bad_channels = list(window_result.bad_channels)
                if game is not None:
                    if bad_channels:
                        game.set_stream_ready(False)
                    elif decision.reason != "warming_up":
                        game.set_stream_ready(True)
                record: dict[str, object] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic": time.monotonic(),
                    "read_monotonic": read_monotonic,
                    "elapsed": time.monotonic() - started,
                    "segment_id": current_segment,
                    "window_start": float(buffer_timestamps[-window_samples]),
                    "window_end": float(buffer_timestamps[-1]),
                    "window_samples": window_samples,
                    "window_span_seconds": float(buffer_timestamps[-1] - buffer_timestamps[-window_samples] + 1.0 / sample_rate),
                    "window_reset_id": window_reset_id,
                    "segment_samples_received": transport.segments[current_segment]["sample_count"],
                    "inference_invoked": not bool(bad_channels),
                    "bad_channels": bad_channels,
                    "model_origin": provenance.get("data_origin", "unspecified"),
                    "model_subject": bundle.subject_id,
                }
                if game_snapshot is not None:
                    record["game"] = {"phase": game_snapshot.phase, "visit_token": game_snapshot.visit_token, "mapping": {label: game_snapshot.command_for(label) for label in bundle.classes}, "control_provenance": provenance}
                if bad_channels:
                    record.update(
                        {
                            "label": decision.label,
                            "confidence": decision.confidence,
                            "probabilities": decision.probabilities,
                            "accepted": decision.accepted,
                            "reason": decision.reason,
                        }
                    )
                    print(f"unknown | bad_signal | bad_channels={bad_channels}")
                    if bars is not None:
                        for bar in bars:
                            bar.set_height(0.0)
                        names_text = "\n".join(
                            ", ".join(bad_channels[index:index + 8])
                            for index in range(0, len(bad_channels), 8)
                        )
                        status_text.set_text(f"结果：unknown | 信号异常\n{names_text}")
                        figure.canvas.draw_idle()
                        figure.canvas.flush_events()
                else:
                    record.update(
                        {
                            "label": decision.label,
                            "confidence": decision.confidence,
                            "probabilities": decision.probabilities,
                            "accepted": decision.accepted,
                            "reason": decision.reason,
                        }
                    )
                    probability_text = " ".join(
                        f"{name}={value:.2f}" for name, value in decision.probabilities.items()
                    )
                    print(
                        f"{decision.label:>10} | confidence={decision.confidence:.2f} | "
                        f"{decision.reason} | {probability_text}"
                    )
                    if (
                        decision.accepted
                        and game is not None
                        and game_snapshot.accepting
                        and time.monotonic() - last_command_time >= args.command_cooldown
                    ):
                        command = game_snapshot.command_for(decision.label)
                        if command and game.submit(command, game_snapshot.visit_token):
                            record["unity"] = {"queued": True, "visit_token": game_snapshot.visit_token}
                            record["command"] = command
                            last_command_time = time.monotonic()
                    if bars is not None:
                        for bar, class_name in zip(bars, bundle.classes):
                            bar.set_height(decision.probabilities[class_name])
                        status_text.set_text(
                            f"结果：{decision.label}  置信度：{decision.confidence:.2f}  {decision.reason}"
                        )
                        figure.canvas.draw_idle()
                        figure.canvas.flush_events()
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
            capture_end = time.monotonic()
    except KeyboardInterrupt:
        capture_error = "KeyboardInterrupt"
        print("用户停止在线分类。")
    except Exception as exception:
        capture_error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        transport.finish(source, capture_end=capture_end, error=capture_error)
        if game is not None:
            game.set_stream_ready(False)
            game.close()
            with decisions_path.open("a", encoding="utf-8") as log_file:
                for event in game.drain_events():
                    log_file.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "unity", "reason": "unity_cleanup", "label": "unknown", **event}, ensure_ascii=False) + "\n")
        if figure is not None:
            plt.close(figure)
    print(f"在线决策日志：{decisions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

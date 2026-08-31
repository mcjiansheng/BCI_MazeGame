"""Classify live LK-Mini motor imagery with smoothing and low-confidence rejection."""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bci_maze.lk_mini.cli import add_source_arguments, source_from_args
from bci_maze.lk_mini.model import load_model_bundle
from bci_maze.lk_mini.online import (
    ProbabilitySmoother,
    RollingEEGBuffer,
    validate_stream_metadata,
)
from bci_maze.lk_mini.quality import assess_signal_quality


COMMAND_MAPPING = {"left_hand": "left", "right_hand": "right", "feet": "up", "tongue": "stop"}


def send_unity(command: str, host: str, port: int, timeout: float = 0.5) -> dict[str, object]:
    payload = json.dumps({"command": command}).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(payload, (host, port))
        try:
            response, _ = sock.recvfrom(2048)
            return json.loads(response.decode("utf-8"))
        except socket.timeout:
            return {"accepted": False, "error": "ack_timeout"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--unity-host")
    parser.add_argument("--unity-port", type=int, default=7777)
    parser.add_argument("--command-cooldown", type=float, default=1.5)
    args = parser.parse_args()

    bundle = load_model_bundle(args.model)
    source = source_from_args(args)
    window_samples = int(round(bundle.config.epoch_seconds * bundle.config.sample_rate))
    step_samples = max(1, int(round(args.step_seconds * bundle.config.sample_rate)))
    buffer = RollingEEGBuffer(len(bundle.channel_names), window_samples + step_samples)
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
        plt.ion()
        figure, axes = plt.subplots(figsize=(8, 5))
        bars = axes.bar(bundle.classes, np.zeros(len(bundle.classes)), color="#4c9be8")
        axes.set_ylim(0.0, 1.0)
        axes.set_ylabel("平滑分类概率")
        axes.set_title("LK-Mini 在线运动想象分类")
        status_text = axes.text(0.5, 0.92, "等待 4 秒数据窗", ha="center", transform=axes.transAxes, fontsize=15)

    samples_since_decision = 0
    last_command_time = -1e9
    try:
        with source, decisions_path.open("a", encoding="utf-8") as log_file:
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
            started = time.monotonic()
            while args.duration <= 0 or time.monotonic() - started < args.duration:
                chunk = source.read(timeout=0.25)
                if not chunk.data.shape[1]:
                    continue
                buffer.append(chunk.data)
                samples_since_decision += chunk.data.shape[1]
                if buffer.sample_count < window_samples or samples_since_decision < step_samples:
                    continue
                samples_since_decision %= step_samples
                window = buffer.latest(window_samples)
                quality = assess_signal_quality(window, sample_rate)
                bad_channels = [name for name, item in zip(bundle.channel_names, quality) if item.status == "bad"]
                record: dict[str, object] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "bad_channels": bad_channels,
                }
                if bad_channels:
                    smoother.reset()
                    record.update({"label": "unknown", "accepted": False, "reason": "bad_signal"})
                    print(f"unknown | bad_signal | bad_channels={bad_channels}")
                    if bars is not None:
                        for bar in bars:
                            bar.set_height(0.0)
                        status_text.set_text(f"结果：unknown  bad_signal  {', '.join(bad_channels)}")
                        figure.canvas.draw_idle()
                        figure.canvas.flush_events()
                else:
                    raw_probabilities = bundle.predict_proba(
                        window, sample_rate, channel_names
                    )[0]
                    decision = smoother.update(raw_probabilities)
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
                        and args.unity_host
                        and time.monotonic() - last_command_time >= args.command_cooldown
                    ):
                        command = COMMAND_MAPPING.get(decision.label)
                        if command and command != "stop":
                            record["unity"] = send_unity(command, args.unity_host, args.unity_port)
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
    except KeyboardInterrupt:
        print("用户停止在线分类。")
    print(f"在线决策日志：{decisions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

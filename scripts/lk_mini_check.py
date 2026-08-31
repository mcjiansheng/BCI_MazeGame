"""Connect to LK-Mini/OpenBCI LSL and inspect live channel signal quality."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bci_maze.lk_mini.cli import add_source_arguments, source_from_args
from bci_maze.lk_mini.quality import assess_signal_quality, format_quality_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--display-seconds", type=float, default=5.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=Path, help="optional raw NPZ output")
    args = parser.parse_args()
    source = source_from_args(args)
    all_data: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []

    figure = axes = lines = None
    sample_rate = int(args.sample_rate)
    channel_names = tuple(args.channel_names)
    capacity = max(1, int(args.display_seconds * sample_rate))
    started = 0.0
    next_quality = 0.0
    try:
        with source:
            sample_rate = int(source.sample_rate)
            channel_names = tuple(source.channel_names)
            if sample_rate <= 0 or not channel_names:
                raise RuntimeError("EEG source did not expose valid stream metadata")
            capacity = max(1, int(args.display_seconds * sample_rate))
            display = np.empty((len(channel_names), 0), dtype=np.float64)
            if not args.no_plot:
                plt.ion()
                figure, axes = plt.subplots(figsize=(12, 7))
                lines = [
                    axes.plot([], [], linewidth=0.8, label=name)[0]
                    for name in channel_names
                ]
                axes.set_title("LK-Mini-EEG16 实时通道检查（显示值按通道错开）")
                axes.set_xlabel("最近时间 / s")
                axes.set_ylabel("幅值 + 通道偏移 / µV")
                axes.legend(loc="upper right", ncols=2)
            started = time.monotonic()
            next_quality = started
            print(
                f"已连接 backend={args.backend}, {sample_rate} Hz, "
                f"通道={channel_names}"
            )
            while time.monotonic() - started < args.duration:
                chunk = source.read(timeout=0.25)
                if not chunk.data.shape[1]:
                    continue
                all_data.append(chunk.data)
                all_timestamps.append(chunk.monotonic_timestamps)
                display = np.concatenate((display, chunk.data), axis=1)[:, -capacity:]
                now = time.monotonic()
                if now >= next_quality and display.shape[1] >= sample_rate * 2:
                    quality = assess_signal_quality(display, sample_rate)
                    print("\n" + format_quality_table(channel_names, quality))
                    next_quality = now + 2.0
                if figure is not None and lines is not None and display.shape[1]:
                    x = (
                        np.arange(display.shape[1]) / sample_rate
                        - display.shape[1] / sample_rate
                    )
                    robust_span = max(50.0, float(np.nanpercentile(np.abs(display), 95)) * 2.5)
                    for index, line in enumerate(lines):
                        line.set_data(x, display[index] + index * robust_span)
                    axes.set_xlim(x[0], 0.0)
                    axes.set_ylim(-robust_span, robust_span * len(lines))
                    figure.canvas.draw_idle()
                    figure.canvas.flush_events()
    except KeyboardInterrupt:
        print("用户停止检查。")

    if not all_data:
        print("未收到任何 EEG 数据。检查 Wi-Fi/LSL、设备 IP、端口和数据流状态。")
        return 2
    continuous = np.concatenate(all_data, axis=1)
    timestamps = np.concatenate(all_timestamps)
    elapsed = max(time.monotonic() - started, 1e-9)
    effective_rate = continuous.shape[1] / elapsed
    rate_error = abs(effective_rate - sample_rate) / sample_rate
    print(
        f"采样核对：收到 {continuous.shape[1]} 点 / {elapsed:.2f}s = "
        f"{effective_rate:.1f} Hz（目标 {sample_rate} Hz）"
    )
    if rate_error > 0.15:
        print("警告：实际接收速率与目标相差超过 15%，请检查设备采样率命令、丢包或 LSL 配置。")
    final_quality = assess_signal_quality(
        continuous[:, -min(continuous.shape[1], capacity) :], sample_rate
    )
    print("\n最终质量摘要\n" + format_quality_table(channel_names, final_quality))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            data=continuous.astype(np.float32),
            timestamps=timestamps,
            sample_rate=np.asarray(sample_rate),
            channel_names=np.asarray(channel_names),
            quality_json=np.asarray(json.dumps([item.to_dict() for item in final_quality])),
        )
        print(f"原始检查数据已保存：{args.output}")
    quality_ok = all(item.status != "bad" for item in final_quality)
    return 0 if quality_ok and rate_error <= 0.15 else 3


if __name__ == "__main__":
    raise SystemExit(main())

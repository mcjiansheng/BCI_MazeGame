"""Connect to LK-Mini/OpenBCI LSL and inspect live channel signal quality."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bci_maze.lk_mini.cli import add_source_arguments, source_from_args
from bci_maze.lk_mini.display import ChannelDisplay
from bci_maze.lk_mini.quality import assess_signal_quality, format_quality_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--display-seconds", type=float, default=5.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=Path, help="optional raw NPZ output")
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    source = source_from_args(args)
    all_data: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_received_timestamps: list[np.ndarray] = []
    all_segment_ids: list[np.ndarray] = []
    received_samples = 0

    plot = None
    quality = None
    sample_rate = int(args.sample_rate)
    channel_names = tuple(args.channel_names)
    capacity = max(1, int(args.display_seconds * sample_rate))
    started = 0.0
    next_quality = 0.0
    capture_error = ""
    current_segment = 0
    recovery_announced = False
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
                labels = channel_names if args.backend == "replay" else tuple(
                    f"CH{channel + 1} {name}"
                    for channel, name in zip(args.channels, channel_names)
                )
                plot = ChannelDisplay(labels)
                plot.figure.suptitle(
                    f"LK-Mini-EEG16 | {len(labels)} 通道 | {args.backend}"
                    + ("（仿真信号）" if args.backend == "synthetic" else "")
                )
            started = time.monotonic()
            deadline = started + args.duration
            next_quality = started
            print(
                f"采集检查启动 backend={args.backend}, {sample_rate} Hz, "
                f"通道={channel_names}"
            )
            while time.monotonic() < deadline:
                if plot is not None:
                    if not plt.fignum_exists(plot.figure.number):
                        break
                    plot.figure.canvas.flush_events()
                if args.no_realtime and received_samples >= int(
                    round(args.duration * sample_rate)
                ):
                    break
                chunk = source.read(timeout=min(0.25, max(0.0, deadline - time.monotonic())))
                recovering = bool(getattr(source, "recovering", False))
                if int(chunk.segment_id) != current_segment or recovering:
                    display = np.empty((len(channel_names), 0), dtype=np.float64)
                    quality = None
                    current_segment = int(chunk.segment_id)
                if recovering and not recovery_announced:
                    if received_samples:
                        print(f"数据流中断，正在恢复连接；下一段编号 {current_segment}。")
                    else:
                        print("正在连接设备，等待首个有效数据块。")
                recovery_announced = recovering
                if not chunk.data.shape[1]:
                    if source.exhausted:
                        break
                    continue
                if not received_samples:
                    print("已收到设备数据，开始持续采集。")
                all_data.append(chunk.data)
                all_timestamps.append(chunk.monotonic_timestamps)
                all_received_timestamps.append(
                    chunk.received_timestamps if chunk.received_timestamps is not None
                    else chunk.monotonic_timestamps
                )
                all_segment_ids.append(np.full(chunk.data.shape[1], current_segment, dtype=np.int64))
                received_samples += chunk.data.shape[1]
                display = np.concatenate((display, chunk.data), axis=1)[:, -capacity:]
                now = time.monotonic()
                if now >= next_quality and display.shape[1] >= sample_rate * 2:
                    quality = assess_signal_quality(display, sample_rate)
                    print("\n" + format_quality_table(channel_names, quality))
                    next_quality = now + 2.0
                if plot is not None and display.shape[1]:
                    plot.update(display, sample_rate, quality)
    except KeyboardInterrupt:
        capture_error = "User interrupted the device check"
        print("用户停止检查。")
    except Exception as exception:
        capture_error = f"{type(exception).__name__}: {exception}"
        print(f"设备检查中断：{capture_error}")
        if all_data:
            print("仅保留中断前已验证的数据；本次检查不通过，不用于正式 MI 训练。")
    finally:
        if plot is not None:
            plt.close(plot.figure)

    if not all_data:
        print("未收到任何 EEG 数据。检查 Wi-Fi/LSL、设备 IP、端口和数据流状态。")
        if args.output and started:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output, data=np.empty((len(channel_names), 0), dtype=np.float32),
                timestamps=np.empty(0), received_timestamps=np.empty(0),
                segment_ids=np.empty(0, dtype=np.int64), sample_rate=np.asarray(sample_rate),
                channel_names=np.asarray(channel_names), segments_json=np.asarray("[]"),
                recovery_stats_json=np.asarray(json.dumps(dict(getattr(source, "recovery_stats", {})))),
                recovery_events_json=np.asarray(json.dumps(list(getattr(source, "recovery_events", [])))),
                capture_error=np.asarray(capture_error or "No EEG samples received"),
                capture_interrupted=np.asarray(True),
            )
        return 2
    continuous = np.concatenate(all_data, axis=1)
    timestamps = np.concatenate(all_timestamps)
    received_timestamps = np.concatenate(all_received_timestamps)
    segment_ids = np.concatenate(all_segment_ids)
    segment_starts = np.r_[0, np.flatnonzero(np.diff(segment_ids)) + 1]
    segment_stops = np.r_[segment_starts[1:], segment_ids.size]
    segments = [
        {"segment_id": int(segment_ids[start]), "start_sample": int(start), "stop_sample": int(stop),
         "start_timestamp": float(timestamps[start]), "stop_timestamp": float(timestamps[stop - 1])}
        for start, stop in zip(segment_starts, segment_stops)
    ]
    recovery_stats = dict(getattr(source, "recovery_stats", {}))
    recovery_events = list(getattr(source, "recovery_events", []))
    elapsed = max(time.monotonic() - started, 1e-9)
    timestamp_span = float(received_timestamps[-1] - received_timestamps[0]) if timestamps.size >= 2 else 0.0
    timestamps_monotonic = bool(timestamps.size < 2 or np.all(np.diff(timestamps) > 0))
    timestamp_deltas = np.diff(timestamps)
    gap_threshold = 2.5 / sample_rate
    gap_count = int(np.count_nonzero((timestamp_deltas > gap_threshold) | (np.diff(segment_ids) != 0)))
    max_gap_seconds = float(timestamp_deltas.max()) if timestamp_deltas.size else 0.0
    effective_rate = (
        (timestamps.size - 1) / timestamp_span
        if timestamps_monotonic and timestamp_span > 0
        else continuous.shape[1] / elapsed
    )
    rate_error = abs(effective_rate - sample_rate) / sample_rate
    wall_rate = continuous.shape[1] / elapsed
    wall_rate_ok = args.backend not in ("brainflow", "lsl") or abs(wall_rate - sample_rate) / sample_rate <= 0.15
    print(
        f"采样核对：收到 {continuous.shape[1]} 点 / {elapsed:.2f}s；"
        f"接收时间估算 {effective_rate:.1f} Hz，整段吞吐 {wall_rate:.1f} 点/s"
        f"（目标 {sample_rate} Hz）"
    )
    print(f"数据分段：{len(segments)} 段；断流 {recovery_stats.get('gaps', gap_count)} 次；成功重连 {recovery_stats.get('reconnects', 0)} 次。")
    if not wall_rate_ok:
        print("警告：整段采集吞吐与目标相差超过 15%，存在启动延迟、停顿或持续速率不符。")
    if rate_error > 0.15:
        print("警告：实际接收速率与目标相差超过 15%，请检查设备采样率命令、丢包或 LSL 配置。")
    if not timestamps_monotonic:
        print("警告：数据时间戳不是严格递增，存在乱序或重复采样时间。")
    if gap_count:
        print(
            f"警告：检测到 {gap_count} 个超过 {gap_threshold * 1000:.1f} ms 的采样间隙；"
            f"最大间隙 {max_gap_seconds * 1000:.1f} ms。"
        )
    final_start = max(int(segment_starts[-1]), continuous.shape[1] - capacity)
    final_quality = assess_signal_quality(continuous[:, final_start:], sample_rate)
    print("\n最终质量摘要\n" + format_quality_table(channel_names, final_quality))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            data=continuous.astype(np.float32),
            timestamps=timestamps,
            received_timestamps=received_timestamps,
            segment_ids=segment_ids,
            segments_json=np.asarray(json.dumps(segments)),
            recovery_stats_json=np.asarray(json.dumps(recovery_stats)),
            recovery_events_json=np.asarray(json.dumps(recovery_events)),
            sample_rate=np.asarray(sample_rate),
            channel_names=np.asarray(channel_names),
            quality_json=np.asarray(json.dumps([item.to_dict() for item in final_quality])),
            effective_sample_rate=np.asarray(effective_rate),
            wall_sample_rate=np.asarray(wall_rate),
            packet_continuity_checked=np.asarray(args.backend == "brainflow"),
            timestamps_monotonic=np.asarray(timestamps_monotonic),
            timestamp_gap_count=np.asarray(gap_count),
            max_timestamp_gap_seconds=np.asarray(max_gap_seconds),
            capture_error=np.asarray(capture_error),
            capture_interrupted=np.asarray(bool(capture_error)),
        )
        print(f"原始检查数据已保存：{args.output}")
    quality_ok = all(item.status != "bad" for item in final_quality)
    return (
        0
        if not capture_error and quality_ok and rate_error <= 0.15 and wall_rate_ok and timestamps_monotonic and gap_count == 0 and not recovery_stats.get("gaps", 0)
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())

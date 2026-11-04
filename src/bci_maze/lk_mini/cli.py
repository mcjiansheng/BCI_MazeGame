"""Shared command-line arguments for LK-Mini scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .source import DEFAULT_CHANNEL_INDICES, DEFAULT_CHANNEL_NAMES, EEGSource, create_source


def parse_channel_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) - 1 for part in value.split(",") if part.strip())
    except ValueError as exception:
        raise argparse.ArgumentTypeError("channels must be comma-separated numbers") from exception
    if not indices or any(index < 0 or index >= 16 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("channels must be unique values from 1 to 16")
    return indices


def parse_channel_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("channel names must be non-empty and unique")
    return names


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("brainflow", "lsl", "replay", "synthetic"),
        default="brainflow",
        help="brainflow direct device, OpenBCI GUI LSL, NPZ replay, or synthetic test data",
    )
    parser.add_argument("--sample-rate", type=int, choices=(250, 500, 1000), default=250)
    parser.add_argument(
        "--channels", type=parse_channel_indices, default=DEFAULT_CHANNEL_INDICES,
        help="1-based device channels (default: all CH1-CH16)",
    )
    parser.add_argument(
        "--channel-names",
        type=parse_channel_names,
        default=DEFAULT_CHANNEL_NAMES,
        help="10-10 electrode names in --channels order (default: " + ",".join(DEFAULT_CHANNEL_NAMES) + ")",
    )
    parser.add_argument("--ip-address", default="192.168.4.1")
    parser.add_argument("--ip-port", type=int, default=12345)
    parser.add_argument("--gain", type=int, choices=(1, 2, 4, 6, 8, 12, 24), default=24)
    parser.add_argument(
        "--auto-reconnect", action=argparse.BooleanOptionalAction, default=True,
        help="isolate and automatically reconnect BrainFlow sessions (default: enabled)",
    )
    parser.add_argument("--connect-timeout", type=float, default=12.0, help="deadline for one BrainFlow connection attempt")
    parser.add_argument("--no-data-timeout", type=float, default=3.0, help="restart BrainFlow after this many seconds without EEG")
    parser.add_argument("--max-retries", type=int, default=0, help="consecutive retries; 0 retries until stopped")
    parser.add_argument("--retry-delay", type=float, default=0.5, help="initial retry delay in seconds")
    parser.add_argument("--max-retry-delay", type=float, default=3.0, help="maximum retry delay in seconds")
    parser.add_argument("--shutdown-timeout", type=float, default=4.0, help="grace period before terminating an unresponsive owned worker")
    parser.add_argument("--device-timeout", type=int, default=5, help="BrainFlow HTTP timeout in seconds")
    parser.add_argument("--lsl-name", help="optional exact LSL stream name; otherwise type=EEG")
    parser.add_argument("--replay", type=Path, help="recording NPZ used by replay backend")
    parser.add_argument("--no-realtime", action="store_true", help="do not pace synthetic/replay data")


def source_from_args(args: argparse.Namespace) -> EEGSource:
    if len(args.channels) != len(args.channel_names):
        raise ValueError("--channels and --channel-names must contain the same number of items")
    return create_source(
        args.backend,
        sample_rate=args.sample_rate,
        channel_indices=args.channels,
        channel_names=args.channel_names,
        ip_address=args.ip_address,
        ip_port=args.ip_port,
        gain=args.gain,
        lsl_name=args.lsl_name,
        replay_path=args.replay,
        realtime=not args.no_realtime,
        auto_reconnect=getattr(args, "auto_reconnect", True),
        connect_timeout=getattr(args, "connect_timeout", 12.0),
        no_data_timeout=getattr(args, "no_data_timeout", 3.0),
        max_retries=getattr(args, "max_retries", 0),
        retry_delay=getattr(args, "retry_delay", 0.5),
        max_retry_delay=getattr(args, "max_retry_delay", 3.0),
        shutdown_timeout=getattr(args, "shutdown_timeout", 4.0),
        device_timeout=getattr(args, "device_timeout", 5),
    )

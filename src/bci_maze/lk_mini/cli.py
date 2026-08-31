"""Shared command-line arguments for LK-Mini scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .source import DEFAULT_CHANNEL_NAMES, EEGSource, create_source


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
    parser.add_argument("--channels", type=parse_channel_indices, default=tuple(range(8)), help="1-based device channels")
    parser.add_argument(
        "--channel-names",
        type=parse_channel_names,
        default=DEFAULT_CHANNEL_NAMES,
        help="10-20 names in exactly the same order as --channels",
    )
    parser.add_argument("--ip-address", default="192.168.4.1")
    parser.add_argument("--ip-port", type=int, default=12345)
    parser.add_argument("--gain", type=int, choices=(1, 2, 4, 6, 8, 12, 24), default=24)
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
    )

"""Simulate streaming inference on offline evaluation trials.

Feeds the evaluation session's raw trials through :class:`OnlineDecoder`
(model-specific causal preprocessing, frozen normalization and hysteresis
policy) and reports window-level accuracy plus command-flip statistics. This
validates the online code path before real hardware arrives.

Optionally forwards decoded commands to the Unity maze over UDP.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.online import HysteresisPolicy, OnlineDecoder, load_label_map
from bci_maze.preprocessing import load_processed_session
from bci_maze.training import norm_stats_to_serializable, prepare_subject_data


def display_path(path: Path) -> str:
    """Show repository-relative paths when possible, absolute paths otherwise."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), default="bcic2a")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--checkpoint", help="Defaults to outputs/checkpoints/<dataset>/se_mhaf_conformer_final/<prefix>NN.pt")
    parser.add_argument("--label-map", help="Defaults to configs/label_maps/<dataset>.json")
    parser.add_argument("--enter-threshold", type=float, default=0.6)
    parser.add_argument("--exit-threshold", type=float, default=0.4)
    parser.add_argument("--send", action="store_true", help="Forward commands to Unity over UDP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prefix = "B" if args.dataset == "bcic2b" else "A"
    label_map_path = Path(args.label_map) if args.label_map else ROOT / "configs" / "label_maps" / f"{args.dataset}.json"
    label_map = load_label_map(label_map_path)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else ROOT / f"outputs/checkpoints/{args.dataset}/se_mhaf_conformer_final/{prefix}{args.subject:02d}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    norm_stats = checkpoint.get("norm_stats")
    if norm_stats is None:
        # Legacy checkpoints predate norm_stats persistence; recompute them
        # with the exact training-time split so inference stays consistent.
        model_name = str(checkpoint["model_name"])
        data = prepare_subject_data(
            ROOT / f"data/processed/{args.dataset}",
            subject=args.subject,
            model_name=model_name,
            validation_size=0.2,
            seed=42 + args.subject,
            exclude_artifacts=True,
            file_prefix=prefix,
        )
        norm_stats = norm_stats_to_serializable(data["norm_stats"])

    session = load_processed_session(
        ROOT / f"data/processed/{args.dataset}", args.subject, "E", file_prefix=prefix
    )
    keep = ~session["artifact"]
    raw_trials = session["raw_x"][keep]
    labels = session["y"][keep]
    if labels.size and (labels.min() < 0 or labels.max() >= label_map.n_classes):
        raise ValueError(
            f"Evaluation labels are outside the configured {label_map.n_classes} classes"
        )
    sfreq = float(session["sfreq"])
    n_channels, n_times = raw_trials.shape[1], raw_trials.shape[2]

    decoder = OnlineDecoder(
        checkpoint_path,
        label_map,
        n_channels=n_channels,
        n_times=n_times,
        sfreq=sfreq,
        device="cpu",
        policy=HysteresisPolicy(args.enter_threshold, args.exit_threshold),
        norm_stats=norm_stats,
    )

    windows: list[dict] = []
    sent_commands = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.send else None
    try:
        for index, (trial, label) in enumerate(zip(raw_trials, labels)):
            decoder.reset_window()
            decoder.push_raw(trial)
            result = decoder.decode_window()
            command = result["command"]
            record = {
                "window_index": index,
                "true_label": int(label),
                "true_class": label_map.labels[int(label)],
                "decoded_label": result["label"],
                "decoded_class": result["label_name"],
                "policy_state": result["state_name"],
                "command": command,
                "confidence": result["confidence"],
            }
            if sock is not None and command is not None:
                payload = json.dumps(
                    {"command": command, "ts": round(time.perf_counter() * 1000.0, 3)}
                ).encode("utf-8")
                sock.sendto(payload, (args.host, args.port))
                sent_commands += 1
                time.sleep(0.15)  # respect Unity's move cooldown
            windows.append(record)
    finally:
        if sock is not None:
            sock.close()

    decoded = np.asarray([window["decoded_label"] for window in windows])
    states = [window["policy_state"] for window in windows]
    flips = sum(1 for previous, current in zip(states, states[1:]) if previous != current)
    accuracy = float((decoded == labels).mean()) if labels.size else 0.0
    report = {
        "dataset": args.dataset,
        "subject": args.subject,
        "checkpoint": display_path(checkpoint_path),
        "label_map": display_path(label_map_path),
        "policy": {"enter_threshold": args.enter_threshold, "exit_threshold": args.exit_threshold},
        "n_windows": int(labels.size),
        "window_accuracy": accuracy,
        "policy_state_flips": flips,
        "sent_commands": sent_commands,
        "note": (
            "Independent evaluation trials are replayed through the online preprocessing "
            "path; this does not simulate device or network timing."
        ),
        "windows": windows,
    }
    output = Path(args.output) if args.output else ROOT / f"outputs/online_replay_{args.dataset}_{prefix}{args.subject:02d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"windows={report['n_windows']} accuracy={accuracy:.4f} "
        f"flips={flips} sent={sent_commands} -> {display_path(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

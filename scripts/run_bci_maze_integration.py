"""Run a real EEG -> classifier -> Unity UDP movement integration test.

This uses the processed BCIC IV 2a evaluation data and the corresponding
SE-MHAF-Final checkpoint. It reproduces the repository's normalization through
``prepare_subject_data`` before sending each model-derived movement to Unity.

The label -> command mapping is loaded from ``configs/label_maps/`` so it can
be reconfigured per subject without code changes. Each UDP payload carries a
send timestamp that Unity echoes back in its acknowledgement, enabling
round-trip latency measurement.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.models import build_model
from bci_maze.online import load_label_map
from bci_maze.training import prepare_subject_data


def display_path(path: Path) -> str:
    """Show repository-relative paths when possible, absolute paths otherwise."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def load_model(checkpoint_path: Path, n_channels: int, n_times: int, n_classes: int) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name", "se_mhaf_conformer_final_logvar")
    model = build_model(model_name, n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), str(model_name)


def send_command(sock: socket.socket, host: str, port: int, command: str, timeout: float) -> dict[str, object]:
    sent_at_ms = time.perf_counter() * 1000.0
    payload = json.dumps({"command": command, "ts": round(sent_at_ms, 3)}).encode("utf-8")
    sock.sendto(payload, (host, port))
    sock.settimeout(timeout)
    try:
        response, _ = sock.recvfrom(2048)
    except socket.timeout:
        return {"ack_received": False, "accepted": False, "rtt_ms": None}
    acknowledgement = json.loads(response.decode("utf-8"))
    return {
        "ack_received": True,
        "accepted": bool(acknowledgement.get("accepted", False)),
        "acknowledged_command": acknowledgement.get("command"),
        "rtt_ms": round(time.perf_counter() * 1000.0 - sent_at_ms, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), default="bcic2a")
    parser.add_argument("--max-trials", type=int, default=32)
    parser.add_argument(
        "--full-replay",
        action="store_true",
        help="Replay every trial instead of stopping at the first accepted movement",
    )
    parser.add_argument("--label-map", help="Path to a label map JSON (defaults to configs/label_maps/<dataset>.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/bci_maze_integration_test.json")
    args = parser.parse_args()

    prefix = "B" if args.dataset == "bcic2b" else "A"
    label_map_path = Path(args.label_map) if args.label_map else ROOT / "configs" / "label_maps" / f"{args.dataset}.json"
    label_map = load_label_map(label_map_path)

    data = prepare_subject_data(
        ROOT / f"data/processed/{args.dataset}",
        subject=args.subject,
        model_name="se_mhaf_conformer_final_logvar" if args.dataset == "bcic2a" else "se_mhaf_conformer_final",
        validation_size=0.2,
        seed=42 + args.subject,
        exclude_artifacts=True,
        file_prefix=prefix,
    )
    test_x = data["test_x"]
    test_y = data["test_y"]
    if test_y.size == 0:
        raise ValueError("The evaluation session has no usable trials")
    if test_y.min() < 0 or test_y.max() >= label_map.n_classes:
        raise ValueError(
            f"Evaluation labels are outside the configured {label_map.n_classes} classes"
        )
    checkpoint = ROOT / f"outputs/checkpoints/{args.dataset}/se_mhaf_conformer_final/{prefix}{args.subject:02d}.pt"
    model, model_name = load_model(checkpoint, test_x.shape[-2], test_x.shape[-1], label_map.n_classes)

    attempts: list[dict[str, object]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, torch.no_grad():
        for index in range(min(args.max_trials, len(test_x))):
            logits = model(torch.from_numpy(test_x[index : index + 1]))
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_label = int(probabilities.argmax().item())
            confidence = float(probabilities[predicted_label].item())
            command = label_map.command_for(predicted_label)
            record: dict[str, object] = {
                "trial_index": index,
                "true_label": int(test_y[index]),
                "true_class": label_map.labels[int(test_y[index])],
                "predicted_label": predicted_label,
                "predicted_class": label_map.labels[predicted_label],
                "confidence": confidence,
                "command": command,
            }
            if command is None:
                record["reason"] = f"{label_map.labels[predicted_label]} maps to no movement"
                attempts.append(record)
                continue

            record.update(send_command(sock, args.host, args.port, command, args.ack_timeout))
            attempts.append(record)
            if record["accepted"] and not args.full_replay:
                break
            # Match Unity's 120 ms anti-bounce window before trying another EEG trial.
            time.sleep(0.15)

    successful = next((attempt for attempt in attempts if attempt.get("accepted")), None)
    rtt_values = [attempt["rtt_ms"] for attempt in attempts if attempt.get("rtt_ms") is not None]
    report = {
        "success": successful is not None,
        "dataset": f"{args.dataset} processed {prefix}{args.subject:02d} evaluation session",
        "checkpoint": display_path(checkpoint),
        "model": model_name,
        "label_map": display_path(label_map_path),
        "label_mapping": {
            label: (command if command is not None else "stop (no movement)")
            for label, command in zip(label_map.labels, label_map.commands)
        },
        "attempts": attempts,
        "successful_movement": successful,
        "rtt_ms_mean": round(sum(rtt_values) / len(rtt_values), 3) if rtt_values else None,
        "rtt_ms_max": max(rtt_values) if rtt_values else None,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

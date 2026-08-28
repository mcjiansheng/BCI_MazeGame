"""Run a real EEG -> classifier -> Unity UDP movement integration test.

This uses the processed BCIC IV 2a A01 evaluation data and the corresponding
SE-MHAF-Final checkpoint. It reproduces the repository's normalization through
``prepare_subject_data`` before sending each model-derived movement to Unity.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import torch

from bci_maze.models import build_model
from bci_maze.training import prepare_subject_data


ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_COMMAND = {0: "left", 1: "right", 2: "up"}
LABEL_NAMES = ("left_hand", "right_hand", "feet", "tongue_stop")


def load_model(checkpoint_path: Path, n_channels: int, n_times: int, n_classes: int) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name", "se_mhaf_conformer_final_logvar")
    model = build_model(model_name, n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def send_command(sock: socket.socket, host: str, port: int, command: str, timeout: float) -> dict[str, object]:
    payload = json.dumps({"command": command}).encode("utf-8")
    sock.sendto(payload, (host, port))
    sock.settimeout(timeout)
    try:
        response, _ = sock.recvfrom(2048)
    except socket.timeout:
        return {"ack_received": False, "accepted": False}
    acknowledgement = json.loads(response.decode("utf-8"))
    return {
        "ack_received": True,
        "accepted": bool(acknowledgement.get("accepted", False)),
        "acknowledged_command": acknowledgement.get("command"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/bci_maze_integration_test.json")
    args = parser.parse_args()

    data = prepare_subject_data(
        ROOT / "data/processed/bcic2a",
        subject=args.subject,
        model_name="se_mhaf_conformer_final_logvar",
        validation_size=0.2,
        seed=42 + args.subject,
        exclude_artifacts=True,
        file_prefix="A",
    )
    test_x = data["test_x"]
    test_y = data["test_y"]
    checkpoint = ROOT / f"outputs/checkpoints/bcic2a/se_mhaf_conformer_final/A{args.subject:02d}.pt"
    model = load_model(checkpoint, test_x.shape[-2], test_x.shape[-1], int(test_y.max()) + 1)

    attempts: list[dict[str, object]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, torch.no_grad():
        for index in range(min(args.max_trials, len(test_x))):
            logits = model(torch.from_numpy(test_x[index : index + 1]))
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_label = int(probabilities.argmax().item())
            confidence = float(probabilities[predicted_label].item())
            command = LABEL_TO_COMMAND.get(predicted_label)
            record: dict[str, object] = {
                "trial_index": index,
                "true_label": int(test_y[index]),
                "true_class": LABEL_NAMES[int(test_y[index])],
                "predicted_label": predicted_label,
                "predicted_class": LABEL_NAMES[predicted_label],
                "confidence": confidence,
                "command": command,
            }
            if command is None:
                record["reason"] = "tongue_stop maps to no movement"
                attempts.append(record)
                continue

            record.update(send_command(sock, args.host, args.port, command, args.ack_timeout))
            attempts.append(record)
            if record["accepted"]:
                break
            # Match Unity's 120 ms anti-bounce window before trying another EEG trial.
            time.sleep(0.15)

    successful = next((attempt for attempt in attempts if attempt.get("accepted")), None)
    report = {
        "success": successful is not None,
        "dataset": "BCIC IV 2a processed A01 evaluation session",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "model": "se_mhaf_conformer_final_logvar",
        "label_mapping": {
            "left_hand": "left",
            "right_hand": "right",
            "feet": "up",
            "tongue": "stop (no movement)",
        },
        "attempts": attempts,
        "successful_movement": successful,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

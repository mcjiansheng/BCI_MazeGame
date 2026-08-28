from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.models import build_model
from bci_maze.training import evaluate, prepare_subject_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly reload and verify final checkpoints")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verified = 0
    for row in payload["results"]:
        config = row["training_config"]
        data = prepare_subject_data(
            config["processed_dir"],
            int(row["subject"]),
            row["model_variant"],
            config["validation_size"],
            config["seed"] + int(row["subject"]),
            config["exclude_artifacts"],
            config["file_prefix"],
        )
        model = build_model(
            row["model_variant"],
            n_channels=int(data["train_x"].shape[-2]),
            n_classes=int(data["test_y"].max() + 1),
        ).to(device)
        checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != row["model_variant"]:
            raise AssertionError(f"Variant mismatch in {row['checkpoint']}")
        model.load_state_dict(checkpoint["model"], strict=True)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(data["test_x"]), torch.from_numpy(data["test_y"])),
            batch_size=64,
        )
        measured = evaluate(model, loader, device)
        if abs(float(measured["accuracy"]) - float(row["accuracy"])) > 1e-12:
            raise AssertionError(
                f"Accuracy mismatch for subject {row['subject']}: "
                f"{measured['accuracy']} != {row['accuracy']}"
            )
        verified += 1
        print(
            f"verified {row['dataset']} subject={int(row['subject']):02d} "
            f"accuracy={measured['accuracy']:.4f}"
        )
    if verified != 9:
        raise AssertionError(f"Expected 9 subjects, verified {verified}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.models import build_model
from bci_maze.training import prepare_subject_data


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totals = {"fused": [], "fbc": [], "mhaf": []}
    for row in payload["results"]:
        config = row["training_config"]
        subject = int(row["subject"])
        data = prepare_subject_data(
            config["processed_dir"],
            subject,
            row["model"],
            config["validation_size"],
            config["seed"] + subject,
            config["exclude_artifacts"],
            config["file_prefix"],
        )
        checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        model = build_model(
            row["model"],
            n_channels=row["n_channels"],
            n_classes=row["n_classes"],
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(data["test_x"]), torch.from_numpy(data["test_y"])),
            batch_size=64,
        )
        predictions = {name: [] for name in totals}
        targets = []
        for x, y in loader:
            fbc, mhaf, scale = model.forward_branches(x.to(device))
            outputs = {"fused": model.fuse_logits(fbc, mhaf, scale), "fbc": fbc, "mhaf": mhaf}
            for name, logits in outputs.items():
                predictions[name].append(logits.argmax(1).cpu().numpy())
            targets.append(y.numpy())
        target = np.concatenate(targets)
        scores = {
            name: float((np.concatenate(values) == target).mean())
            for name, values in predictions.items()
        }
        for name, score in scores.items():
            totals[name].append(score)
        print(f"{config['file_prefix']}{subject:02d} " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    print("mean " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in totals.items()))


if __name__ == "__main__":
    main()

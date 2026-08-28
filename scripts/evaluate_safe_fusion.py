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
def predict(model, x: np.ndarray, branch: str | None, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=64)
    outputs = []
    for (batch,) in loader:
        batch = batch.to(device)
        if branch == "mhaf":
            logits = model.forward_branches(batch)[1]
        else:
            logits = model(batch)
        outputs.append(logits.float().cpu())
    return torch.cat(outputs).softmax(1).numpy()


def accuracy(probability: np.ndarray, target: np.ndarray) -> float:
    return float((probability.argmax(1) == target).mean())


def rows_by_subject(path: str, model_name: str) -> dict[int, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(row["subject"]): row
        for row in payload["results"]
        if row["model"] == model_name
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--v2", required=True)
    args = parser.parse_args()
    fbc_rows = rows_by_subject(args.baseline, "fbcnet")
    v2_rows = rows_by_subject(args.v2, "se_mhaf_conformer_v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grids = np.linspace(0.0, 0.5, 11)
    totals = {"fbc": [], "fixed_0.1": [], "fixed_0.2": [], "selected": []}
    for subject in sorted(fbc_rows):
        fbc_row, v2_row = fbc_rows[subject], v2_rows[subject]
        config = fbc_row["training_config"]
        file_prefix = config.get("file_prefix", "B" if args.dataset == "bcic2b" else "A")
        data = prepare_subject_data(
            config["processed_dir"], subject, "fbcnet", config["validation_size"],
            config["seed"] + subject, config["exclude_artifacts"], file_prefix,
        )
        n_channels = int(data["train_x"].shape[-2])
        n_classes = int(max(data["train_y"].max(), data["test_y"].max()) + 1)
        fbc = build_model("fbcnet", n_channels=n_channels, n_classes=n_classes).to(device)
        fbc.load_state_dict(torch.load(fbc_row["checkpoint"], map_location="cpu", weights_only=True)["model"])
        v2 = build_model("se_mhaf_conformer_v2", n_channels=n_channels, n_classes=n_classes).to(device)
        v2.load_state_dict(torch.load(v2_row["checkpoint"], map_location="cpu", weights_only=True)["model"])
        fbc.eval(); v2.eval()
        fbc_val = predict(fbc, data["validation_x"], None, device)
        fbc_test = predict(fbc, data["test_x"], None, device)
        mhaf_val = predict(v2, data["validation_x"], "mhaf", device)
        mhaf_test = predict(v2, data["test_x"], "mhaf", device)
        val_scores = [accuracy((1-alpha)*fbc_val + alpha*mhaf_val, data["validation_y"]) for alpha in grids]
        best_val = max(val_scores)
        # Conservative tie break: preserve the baseline unless fusion is better.
        alpha = float(grids[next(i for i, score in enumerate(val_scores) if score == best_val)])
        scores = {
            "fbc": accuracy(fbc_test, data["test_y"]),
            "fixed_0.1": accuracy(0.9*fbc_test + 0.1*mhaf_test, data["test_y"]),
            "fixed_0.2": accuracy(0.8*fbc_test + 0.2*mhaf_test, data["test_y"]),
            "selected": accuracy((1-alpha)*fbc_test + alpha*mhaf_test, data["test_y"]),
        }
        for name, score in scores.items(): totals[name].append(score)
        print(f"{file_prefix}{subject:02d} alpha={alpha:.2f} val={best_val:.4f} " + " ".join(f"{k}={v:.4f}" for k,v in scores.items()))
    print("mean " + " ".join(f"{k}={np.mean(v):.4f}" for k,v in totals.items()))


if __name__ == "__main__":
    main()

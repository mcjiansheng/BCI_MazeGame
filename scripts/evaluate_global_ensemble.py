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


def result_rows(path: str) -> dict[str, dict[int, dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    output: dict[str, dict[int, dict]] = {}
    for row in payload["results"]:
        output.setdefault(row["model"], {})[int(row["subject"])] = row
    return output


@torch.no_grad()
def probabilities(model, x: np.ndarray, device, branch: str | None = None) -> np.ndarray:
    batches = []
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=64):
        batch = batch.to(device)
        logits = model.forward_branches(batch)[1] if branch == "mhaf" else model(batch)
        batches.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(batches)


def load_model(row: dict, name: str, n_channels: int, n_classes: int, device):
    model = build_model(name, n_channels=n_channels, n_classes=n_classes).to(device)
    state = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state)
    return model.eval()


def acc(probability: np.ndarray, target: np.ndarray) -> float:
    return float((probability.argmax(1) == target).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    baseline = result_rows(args.baseline)
    v2 = result_rows(args.v2)["se_mhaf_conformer_v2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cached = []
    for subject in sorted(baseline["fbcnet"]):
        fbc_row = baseline["fbcnet"][subject]
        eeg_row = baseline["eeg_conformer"][subject]
        config = fbc_row["training_config"]
        prefix = config.get("file_prefix", "B" if args.dataset == "bcic2b" else "A")
        common = (
            config["processed_dir"], subject, config["validation_size"],
            config["seed"] + subject, config["exclude_artifacts"], prefix,
        )
        fbc_data = prepare_subject_data(common[0], common[1], "fbcnet", *common[2:])
        eeg_data = prepare_subject_data(common[0], common[1], "eeg_conformer", *common[2:])
        n_channels = int(fbc_data["train_x"].shape[-2])
        n_classes = int(fbc_data["test_y"].max() + 1)
        fbc = load_model(fbc_row, "fbcnet", n_channels, n_classes, device)
        eeg = load_model(eeg_row, "eeg_conformer", n_channels, n_classes, device)
        mhaf = load_model(v2[subject], "se_mhaf_conformer_v2", n_channels, n_classes, device)
        cached.append({
            "subject": subject,
            "target_val": fbc_data["validation_y"],
            "target_test": fbc_data["test_y"],
            "fbc_val": probabilities(fbc, fbc_data["validation_x"], device),
            "fbc_test": probabilities(fbc, fbc_data["test_x"], device),
            "eeg_val": probabilities(eeg, eeg_data["validation_x"], device),
            "eeg_test": probabilities(eeg, eeg_data["test_x"], device),
            "mhaf_val": probabilities(mhaf, fbc_data["validation_x"], device, "mhaf"),
            "mhaf_test": probabilities(mhaf, fbc_data["test_x"], device, "mhaf"),
        })

    candidates = []
    count = round(1 / args.step)
    for fbc_units in range(count + 1):
        for eeg_units in range(count - fbc_units + 1):
            mhaf_units = count - fbc_units - eeg_units
            weights = np.asarray([fbc_units, eeg_units, mhaf_units], dtype=float) / count
            validation = np.mean([
                acc(weights[0]*row["fbc_val"] + weights[1]*row["eeg_val"] + weights[2]*row["mhaf_val"], row["target_val"])
                for row in cached
            ])
            candidates.append((validation, weights))
    # Prefer more FBC weight when validation accuracy ties.
    validation, weights = max(candidates, key=lambda item: (item[0], item[1][0]))
    print(f"selected validation={validation:.4f} weights fbc={weights[0]:.2f} eeg={weights[1]:.2f} mhaf={weights[2]:.2f}")
    test_scores = []
    for row in cached:
        probability = weights[0]*row["fbc_test"] + weights[1]*row["eeg_test"] + weights[2]*row["mhaf_test"]
        score = acc(probability, row["target_test"])
        test_scores.append(score)
        print(f"subject={row['subject']:02d} accuracy={score:.4f}")
    print(f"test mean={np.mean(test_scores):.4f} std={np.std(test_scores):.4f}")


if __name__ == "__main__":
    main()

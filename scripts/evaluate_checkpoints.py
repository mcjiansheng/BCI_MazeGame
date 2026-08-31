"""Re-evaluate saved checkpoints on their evaluation session (no retraining).

Loads every subject checkpoint of the requested models, rebuilds the exact
model from the checkpoint metadata, reproduces the training-time data split
normalization, and reports per-subject and mean accuracy / macro-F1 / kappa
on the held-out E session.
"""

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


CHECKPOINT_DIRS = {
    "bcic2a": {
        "eeg_conformer": "eeg_conformer",
        "fbcnet": "fbcnet",
        "se_mhaf_conformer": "se_mhaf_conformer",
        "se_mhaf_conformer_v2": "bcic2a/se_mhaf_conformer_v2",
        "se_mhaf_conformer_final": "bcic2a/se_mhaf_conformer_final",
    },
    "bcic2b": {
        "eeg_conformer": "bcic2b/eeg_conformer",
        "fbcnet": "bcic2b/fbcnet",
        "se_mhaf_conformer": "bcic2b/se_mhaf_conformer",
        "se_mhaf_conformer_v2": "bcic2b/se_mhaf_conformer_v2",
        "se_mhaf_conformer_final": "bcic2b/se_mhaf_conformer_final",
    },
}
DATASET_CLASS_COUNTS = {"bcic2a": 4, "bcic2b": 2}


def evaluate_subject(dataset: str, model_key: str, subject: int, device: torch.device) -> dict:
    prefix = "B" if dataset == "bcic2b" else "A"
    checkpoint_dir = ROOT / "outputs" / "checkpoints" / CHECKPOINT_DIRS[dataset][model_key]
    checkpoint_path = checkpoint_dir / f"{prefix}{subject:02d}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = str(checkpoint.get("model_name", model_key))

    data = prepare_subject_data(
        ROOT / f"data/processed/{dataset}",
        subject=subject,
        model_name=model_name,
        validation_size=0.2,
        seed=42 + subject,
        exclude_artifacts=True,
        file_prefix=prefix,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data["test_x"]), torch.from_numpy(data["test_y"])),
        batch_size=64,
        shuffle=False,
    )
    model = build_model(
        model_name,
        n_channels=data["test_x"].shape[-2],
        n_times=data["test_x"].shape[-1],
        # The output width is a dataset property. Inferring it from one
        # artifact-filtered evaluation split can drop an absent highest class
        # and make an otherwise valid checkpoint impossible to load.
        n_classes=DATASET_CLASS_COUNTS[dataset],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = evaluate(model, test_loader, device)
    del model
    return {
        "model": model_key,
        "model_name": model_name,
        "subject": subject,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "validation_accuracy": checkpoint.get("validation_accuracy"),
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), required=True)
    parser.add_argument("--models", default="all", help="Comma-separated model keys or 'all'")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_keys = (
        list(CHECKPOINT_DIRS[args.dataset]) if args.models == "all" else args.models.split(",")
    )

    results = []
    for model_key in model_keys:
        if model_key not in CHECKPOINT_DIRS[args.dataset]:
            raise ValueError(f"Unknown model {model_key!r}")
        for subject in range(1, 10):
            row = evaluate_subject(args.dataset, model_key, subject, device)
            print(
                f"{model_key} {row['subject']:02d} "
                f"acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} kappa={row['kappa']:.4f}",
                flush=True,
            )
            results.append(row)

    summary = {}
    for model_key in model_keys:
        rows = [row for row in results if row["model"] == model_key]
        accuracies = [row["accuracy"] for row in rows]
        summary[model_key] = {
            "n_subjects": len(rows),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "per_subject": {row["subject"]: row["accuracy"] for row in rows},
        }

    payload = {"dataset": args.dataset, "results": results, "summary": summary}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Mean accuracy per model ===")
    for model_key, stats in summary.items():
        print(f"{model_key:28s} {stats['mean_accuracy'] * 100:6.2f}%  (n={stats['n_subjects']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

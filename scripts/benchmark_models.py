from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.training import TrainConfig, save_results, summarize_results, train_subject


MODEL_DEFAULTS = {
    "eeg_conformer": {"epochs": 200, "patience": 40, "batch_size": 64, "learning_rate": 2e-4},
    "fbcnet": {"epochs": 500, "patience": 100, "batch_size": 16, "learning_rate": 1e-3},
    "se_mhaf_conformer": {"epochs": 200, "patience": 40, "batch_size": 64, "learning_rate": 2e-4},
    "se_mhaf_conformer_v2": {"epochs": 500, "patience": 100, "batch_size": 16, "learning_rate": 1e-3},
    "se_mhaf_conformer_v3": {
        "epochs": 1000,
        "patience": 200,
        "batch_size": 72,
        "learning_rate": 2e-4,
        "augment": True,
    },
    "se_mhaf_conformer_v3_raw": {
        "epochs": 1000,
        "patience": 200,
        "batch_size": 72,
        "learning_rate": 2e-4,
        "augment": True,
    },
    "se_mhaf_conformer_v3_compact": {
        "epochs": 500,
        "patience": 100,
        "batch_size": 64,
        "learning_rate": 2e-4,
        "augment": True,
    },
}


def parse_list(value: str, cast=str):
    if value.lower() == "all" and cast is int:
        return list(range(1, 10))
    return [cast(item.strip()) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare BCIC2a/BCIC2b models")
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), default="bcic2a")
    parser.add_argument(
        "--models",
        default="eeg_conformer,fbcnet,se_mhaf_conformer",
        help="Comma-separated model names",
    )
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--epochs", type=int, help="Override model-specific default")
    parser.add_argument("--patience", type=int, help="Override model-specific default")
    parser.add_argument("--batch-size", type=int, help="Override model-specific default")
    parser.add_argument("--lr", type=float, help="Override model-specific default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir")
    parser.add_argument("--output")
    parser.add_argument("--include-artifacts", action="store_true")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--augment", dest="augment", action="store_true")
    augmentation.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    processed_dir = args.processed_dir or f"data/processed/{args.dataset}"
    output = args.output or f"outputs/{args.dataset}_benchmark.json"

    results = []
    for model in parse_list(args.models):
        normalized_model = model.lower().replace("-", "_")
        if normalized_model not in MODEL_DEFAULTS:
            raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODEL_DEFAULTS)}")
        defaults = MODEL_DEFAULTS[normalized_model]
        config = TrainConfig(
            dataset_name=args.dataset,
            file_prefix="B" if args.dataset == "bcic2b" else "A",
            processed_dir=processed_dir,
            output_dir=str(Path(output).parent),
            epochs=args.epochs or defaults["epochs"],
            patience=args.patience or defaults["patience"],
            batch_size=args.batch_size or defaults["batch_size"],
            learning_rate=args.lr or defaults["learning_rate"],
            seed=args.seed,
            exclude_artifacts=not args.include_artifacts,
            augment=defaults.get("augment", False) if args.augment is None else args.augment,
            amp=not args.no_amp,
        )
        for subject in parse_list(args.subjects, int):
            results.append(train_subject(normalized_model, subject, config))
            save_results(results, config, output)
    print(summarize_results(results))


if __name__ == "__main__":
    main()

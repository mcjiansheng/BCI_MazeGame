"""Pretrain 16-channel SE-MHAF on BCIC2a T sessions or evaluate frozen E sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bci_maze.lk_mini.semhaf import load_semhaf_bundle, save_semhaf_bundle
from bci_maze.lk_mini.semhaf_training import TrainingConfig, evaluate_public, pretrain_public, seed_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/bcic2a"))
    parser.add_argument("--train-subjects", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--validation-subjects", nargs="+", type=int, default=[8, 9])
    parser.add_argument("--output", type=Path, default=Path("outputs/pretrained/lk_semhaf16.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluate-only", type=Path, help="Load a frozen model and evaluate E sessions; no training")
    parser.add_argument("--evaluation-subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--evaluation-output", type=Path, help="Separate final E-session JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(args.epochs, args.patience, args.batch_size, args.lr, args.weight_decay, args.seed, args.cpu_threads)
    seed_training(config)
    if args.evaluate_only:
        bundle = load_semhaf_bundle(args.evaluate_only, device=args.device)
        report = evaluate_public(bundle, args.processed_dir, args.evaluation_subjects, args.batch_size)
        report["checkpoint"] = str(args.evaluate_only.resolve())
        destination = args.evaluation_output or args.evaluate_only.with_suffix(".evaluation.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Evaluation: {destination}")
    else:
        bundle = pretrain_public(args.processed_dir, args.train_subjects, args.validation_subjects, config, device=args.device)
        destination = save_semhaf_bundle(bundle, args.output)
        print(f"Checkpoint: {destination}")
        print(json.dumps({"best_epoch": bundle.training_report["best_epoch"], "validation": bundle.training_report["validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

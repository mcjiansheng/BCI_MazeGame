"""Calibrate a public 16-channel SE-MHAF checkpoint with labeled personal recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bci_maze.lk_mini.semhaf import save_semhaf_bundle
from bci_maze.lk_mini.semhaf_training import TrainingConfig, finetune_personal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path, help="Chronological order; the last block is final holdout")
    parser.add_argument("--pretrained", type=Path, default=Path("outputs/pretrained/lk_semhaf16.pt"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--holdout-session", help="Session ID reserved exclusively for final before/after evaluation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(args.epochs, args.patience, args.batch_size, args.lr, args.weight_decay, args.seed, args.cpu_threads)
    bundle = finetune_personal(args.pretrained, args.recordings, args.subject, config,
                              device=args.device, holdout_session=args.holdout_session)
    destination = save_semhaf_bundle(bundle, args.output or Path("outputs/subject_models") / args.subject / "semhaf16.pt")
    print(json.dumps({key: bundle.training_report[key] for key in (
        "data_origin", "before_finetuning", "after_finetuning", "calibration_passed", "human_mi_verified",
    )}, ensure_ascii=False, indent=2))
    print(f"Checkpoint: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.models import build_model
from bci_maze.training import (
    TrainConfig,
    evaluate,
    prepare_subject_data,
    save_results,
    seed_everything,
    segment_reconstruction,
    summarize_results,
)


def parse_subjects(value: str) -> list[int]:
    return list(range(1, 10)) if value.lower() == "all" else [int(x) for x in value.split(",")]


def baseline_rows(path: str) -> dict[int, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(row["subject"]): row
        for row in payload["results"]
        if row["model"] == "fbcnet"
    }


def v2_rows(path: str | None) -> dict[int, dict]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(row["subject"]): row
        for row in payload["results"]
        if row["model"] == "se_mhaf_conformer_v2"
    }


def load_temporal_expert(model: nn.Module, checkpoint_path: str) -> int:
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    destination = model.state_dict()
    prefixes = {
        "branches.": "temporal_branches.",
        "fusion.": "temporal_fusion.",
        "encoder.": "temporal_encoder.",
        "pool.": "temporal_pool.",
        "mhaf_classifier.": "temporal_classifier.",
    }
    transferred = 0
    for key, value in source.items():
        if key == "position":
            destination["temporal_position"] = value
            transferred += 1
            continue
        for source_prefix, destination_prefix in prefixes.items():
            if key.startswith(source_prefix):
                destination[destination_prefix + key[len(source_prefix):]] = value
                transferred += 1
                break
    model.load_state_dict(destination)
    if transferred == 0:
        raise RuntimeError(f"No temporal expert weights found in {checkpoint_path}")
    with torch.no_grad():
        model.expert_mix_logits.copy_(torch.tensor([-6.0, 6.0]))
    return transferred


def train_subject(
    subject: int,
    baseline: dict,
    temporal_expert: dict | None,
    config: TrainConfig,
    min_validation_gain: float,
    logvar_only: bool,
) -> dict:
    seed_everything(config.seed + subject)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    data = prepare_subject_data(
        config.processed_dir,
        subject,
        "se_mhaf_conformer_final",
        config.validation_size,
        config.seed + subject,
        config.exclude_artifacts,
        config.file_prefix,
    )
    tensors = {
        key: torch.from_numpy(data[key])
        for key in ("train_x", "train_y", "validation_x", "validation_y", "test_x", "test_y")
    }
    loaders = {
        "train": DataLoader(
            TensorDataset(tensors["train_x"], tensors["train_y"]),
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            TensorDataset(tensors["validation_x"], tensors["validation_y"]),
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            TensorDataset(tensors["test_x"], tensors["test_y"]),
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        ),
    }
    n_channels = int(data["train_x"].shape[-2])
    n_classes = int(max(data["train_y"].max(), data["test_y"].max()) + 1)
    model_name = "se_mhaf_conformer_final_logvar" if logvar_only else "se_mhaf_conformer_final"
    model = build_model(model_name, n_channels=n_channels, n_classes=n_classes).to(device)
    backbone = torch.load(baseline["checkpoint"], map_location="cpu", weights_only=True)
    model.fbc_branch.load_state_dict(backbone["model"])
    transferred_weights = 0
    if temporal_expert is not None and not logvar_only:
        transferred_weights = load_temporal_expert(model, temporal_expert["checkpoint"])
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(config.seed + subject)
    baseline_validation = evaluate(model, loaders["validation"], device)["accuracy"]
    best_validation = float(baseline_validation)
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    progress_validation = float(baseline_validation)
    if temporal_expert is not None and not logvar_only:
        temporal_qualified = False
        for alpha in np.linspace(0.05, 0.5, 10):
            with torch.no_grad():
                model.residual_scale.fill_(float(np.arctanh(alpha)))
            calibrated = float(evaluate(model, loaders["validation"], device)["accuracy"])
            if calibrated >= baseline_validation + min_validation_gain and calibrated > best_validation:
                temporal_qualified = True
                best_validation = calibrated
                best_epoch = -1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        with torch.no_grad():
            model.residual_scale.zero_()
            # Validation routes each subject to the expert with evidence of
            # complementarity. This avoids diluting the data-efficient logvar
            # path on 2a or the transferred temporal path on 2b.
            route = torch.tensor([-6.0, 6.0] if temporal_qualified else [6.0, -6.0])
            model.expert_mix_logits.copy_(route)
    without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            if config.augment:
                aug_x, aug_y = segment_reconstruction(
                    data["train_x"], data["train_y"], max(1, x.shape[0] // n_classes), rng=rng
                )
                x = torch.cat((x, torch.from_numpy(aug_x)), dim=0)
                y = torch.cat((y, torch.from_numpy(aug_y)), dim=0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fbc, residual, scale = model.forward_branches(x)
                fused = model.fuse_logits(fbc, residual, scale)
                loss = criterion(fused, y) + 0.25 * criterion(residual, y)
                loss = loss + 0.005 * scale.square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        validation = evaluate(model, loaders["validation"], device)
        qualifies = validation["accuracy"] >= baseline_validation + min_validation_gain
        if validation["accuracy"] > progress_validation:
            progress_validation = float(validation["accuracy"])
            without_improvement = 0
        else:
            without_improvement += 1
        if qualifies and validation["accuracy"] > best_validation:
            best_validation = float(validation["accuracy"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"final {config.file_prefix}{subject:02d} epoch={epoch:03d} "
                f"val={validation['accuracy']:.4f} baseline={baseline_validation:.4f} "
                f"best={best_validation:.4f}",
                flush=True,
            )
        if without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device)
    checkpoint_dir = Path(config.output_dir) / "checkpoints" / config.dataset_name / "se_mhaf_conformer_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{config.file_prefix}{subject:02d}.pt"
    torch.save(
        {
            "model": best_state,
            "model_name": model_name,
            "logvar_only": logvar_only,
            "subject": subject,
            "config": asdict(config),
            "backbone_checkpoint": baseline["checkpoint"],
            "temporal_expert_checkpoint": temporal_expert["checkpoint"] if temporal_expert else None,
            "baseline_validation_accuracy": baseline_validation,
            "validation_accuracy": best_validation,
            "test_metrics": test_metrics,
        },
        checkpoint,
    )
    result = {
        "model": "se_mhaf_conformer_final",
        "model_variant": model_name,
        "logvar_only": logvar_only,
        "dataset": config.dataset_name,
        "subject": subject,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "train_trials": int(data["train_y"].size),
        "validation_trials": int(data["validation_y"].size),
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "baseline_validation_accuracy": float(baseline_validation),
        "validation_accuracy": best_validation,
        **test_metrics,
        "excluded_train_artifacts": int(data["excluded_train_artifacts"]),
        "excluded_test_artifacts": int(data["excluded_test_artifacts"]),
        "duration_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint": str(checkpoint),
        "backbone_checkpoint": baseline["checkpoint"],
        "temporal_expert_checkpoint": temporal_expert["checkpoint"] if temporal_expert else None,
        "transferred_temporal_weights": transferred_weights,
        "min_validation_gain": min_validation_gain,
        "training_config": asdict(config),
    }
    print(
        f"final {config.file_prefix}{subject:02d} test={result['accuracy']:.4f} "
        f"best_epoch={best_epoch}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("bcic2a", "bcic2b"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--v2", help="V2 result JSON used to initialize the temporal expert")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--logvar-only", action="store_true")
    parser.add_argument(
        "--min-validation-gain",
        type=float,
        default=0.05,
        help="Minimum absolute validation gain required to enable the residual path",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prefix = "B" if args.dataset == "bcic2b" else "A"
    config = TrainConfig(
        dataset_name=args.dataset,
        file_prefix=prefix,
        processed_dir=f"data/processed/{args.dataset}",
        output_dir=str(Path(args.output).parent),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment=not args.no_augment,
    )
    baselines = baseline_rows(args.baseline)
    temporal_experts = v2_rows(args.v2)
    results = []
    for subject in parse_subjects(args.subjects):
        results.append(
            train_subject(
                subject,
                baselines[subject],
                temporal_experts.get(subject),
                config,
                args.min_validation_gain,
                args.logvar_only,
            )
        )
        save_results(results, config, args.output)
    print(summarize_results(results))


if __name__ == "__main__":
    main()

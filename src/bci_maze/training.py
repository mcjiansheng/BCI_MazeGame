"""Subject-dependent BCIC2a training and evaluation utilities."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model
from .preprocessing import load_processed_session, make_filter_bank


FILTER_BANK_MODELS = {
    "fbcnet",
    "se_mhaf_conformer_v2",
    "semhaf_conformer_v2",
    "se_mhaf_conformer_final",
    "semhaf_conformer_final",
    "se_mhaf_conformer_final_logvar",
    "semhaf_conformer_final_logvar",
}
RAW_BROADBAND_MODELS = {
    "se_mhaf_conformer_paper",
    "se_mhaf_conformer_v3_raw",
    "se_mhaf_conformer_v3_compact",
    "semhaf_conformer_v3_compact",
}


def uses_filter_bank(model_name: str) -> bool:
    return model_name.lower().replace("-", "_") in FILTER_BANK_MODELS


def uses_raw_broadband(model_name: str) -> bool:
    return model_name.lower().replace("-", "_") in RAW_BROADBAND_MODELS


@dataclass
class TrainConfig:
    dataset_name: str = "bcic2a"
    file_prefix: str = "A"
    processed_dir: str = "data/processed/bcic2a"
    output_dir: str = "outputs"
    epochs: int = 100
    patience: int = 20
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    validation_size: float = 0.2
    seed: int = 42
    exclude_artifacts: bool = True
    augment: bool = False
    amp: bool = True
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalize_broadband(
    train_x: np.ndarray, test_x: np.ndarray, fit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x[fit_indices].mean(axis=(0, 2), keepdims=True)
    std = train_x[fit_indices].std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    return ((train_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def _normalize_filter_bank(
    train_x: np.ndarray, test_x: np.ndarray, fit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x[fit_indices].mean(axis=(0, 3), keepdims=True)
    std = train_x[fit_indices].std(axis=(0, 3), keepdims=True).clip(min=1e-6)
    return ((train_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def prepare_subject_data(
    processed_dir: str | Path,
    subject: int,
    model_name: str,
    validation_size: float,
    seed: int,
    exclude_artifacts: bool,
    file_prefix: str = "A",
) -> dict[str, np.ndarray]:
    training = load_processed_session(processed_dir, subject, "T", file_prefix=file_prefix)
    evaluation = load_processed_session(processed_dir, subject, "E", file_prefix=file_prefix)
    train_keep = ~training["artifact"] if exclude_artifacts else np.ones_like(training["artifact"], bool)
    test_keep = ~evaluation["artifact"] if exclude_artifacts else np.ones_like(evaluation["artifact"], bool)
    source_key = (
        "raw_x"
        if uses_filter_bank(model_name) or uses_raw_broadband(model_name)
        else "x"
    )
    if source_key not in training or source_key not in evaluation:
        raise KeyError(
            f"Processed data has no {source_key!r}; rerun preprocessing with --overwrite"
        )
    train_x, train_y = training[source_key][train_keep], training["y"][train_keep]
    test_x, test_y = evaluation[source_key][test_keep], evaluation["y"][test_keep]

    indices = np.arange(train_y.size)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        stratify=train_y,
    )
    if uses_filter_bank(model_name):
        train_x = make_filter_bank(train_x)
        test_x = make_filter_bank(test_x)
        train_x, test_x = _normalize_filter_bank(train_x, test_x, train_indices)
    else:
        train_x, test_x = _normalize_broadband(train_x, test_x, train_indices)
        train_x = train_x[:, None, :, :]
        test_x = test_x[:, None, :, :]
    return {
        "train_x": train_x[train_indices],
        "train_y": train_y[train_indices],
        "validation_x": train_x[validation_indices],
        "validation_y": train_y[validation_indices],
        "test_x": test_x,
        "test_y": test_y,
        "excluded_train_artifacts": np.asarray(int(training["artifact"].sum())),
        "excluded_test_artifacts": np.asarray(int(evaluation["artifact"].sum())),
    }


def segment_reconstruction(
    x: np.ndarray,
    y: np.ndarray,
    samples_per_class: int,
    segments: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """EEG-Conformer's segmentation-and-reconstruction augmentation."""
    rng = rng or np.random.default_rng()
    segment_size = x.shape[-1] // segments
    augmented_x, augmented_y = [], []
    for class_id in np.unique(y):
        candidates = np.flatnonzero(y == class_id)
        class_output = np.empty((samples_per_class, *x.shape[1:]), dtype=np.float32)
        for sample in range(samples_per_class):
            for segment in range(segments):
                source = rng.choice(candidates)
                start = segment * segment_size
                stop = x.shape[-1] if segment == segments - 1 else (segment + 1) * segment_size
                class_output[sample, ..., start:stop] = x[source, ..., start:stop]
        augmented_x.append(class_output)
        augmented_y.append(np.full(samples_per_class, class_id, dtype=np.int64))
    return np.concatenate(augmented_x), np.concatenate(augmented_y)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    predictions, targets = [], []
    n_classes = None
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        n_classes = logits.shape[1]
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    y_true, y_pred = np.concatenate(targets), np.concatenate(predictions)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).tolist(),
        "n_trials": int(y_true.size),
    }


def train_subject(model_name: str, subject: int, config: TrainConfig) -> dict[str, object]:
    seed_everything(config.seed + subject)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    data = prepare_subject_data(
        config.processed_dir,
        subject,
        model_name,
        config.validation_size,
        config.seed + subject,
        config.exclude_artifacts,
        config.file_prefix,
    )
    tensor_data = {
        key: torch.from_numpy(value)
        for key, value in data.items()
        if key in {"train_x", "train_y", "validation_x", "validation_y", "test_x", "test_y"}
    }
    loaders = {
        "train": DataLoader(
            TensorDataset(tensor_data["train_x"], tensor_data["train_y"]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            TensorDataset(tensor_data["validation_x"], tensor_data["validation_y"]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            TensorDataset(tensor_data["test_x"], tensor_data["test_y"]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        ),
    }

    n_channels = int(data["train_x"].shape[-2])
    n_times = int(data["train_x"].shape[-1])
    n_classes = int(max(data["train_y"].max(), data["test_y"].max()) + 1)
    model = build_model(
        model_name,
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    augmentation_rng = np.random.default_rng(config.seed + subject)
    best_state, best_validation, best_epoch, epochs_without_improvement = None, -1.0, 0, 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            if config.augment and not uses_filter_bank(model_name):
                aug_x, aug_y = segment_reconstruction(
                    data["train_x"],
                    data["train_y"],
                    max(1, x.shape[0] // 4),
                    rng=augmentation_rng,
                )
                x = torch.cat((x, torch.from_numpy(aug_x)), dim=0)
                y = torch.cat((y, torch.from_numpy(aug_y)), dim=0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if model_name in {"se_mhaf_conformer_v2", "semhaf_conformer_v2"}:
                    fbc_logits, mhaf_logits, scale = model.forward_branches(x)
                    fused_logits = model.fuse_logits(fbc_logits, mhaf_logits, scale)
                    loss = (
                        criterion(fused_logits, y)
                        + 0.25 * criterion(fbc_logits, y)
                        + 0.25 * criterion(mhaf_logits, y)
                    )
                else:
                    loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        validation = evaluate(model, loaders["validation"], device)
        if validation["accuracy"] > best_validation:
            best_validation = float(validation["accuracy"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{model_name} {config.file_prefix}{subject:02d} epoch={epoch:03d} "
                f"val_acc={validation['accuracy']:.4f} best={best_validation:.4f}",
                flush=True,
            )
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device)
    checkpoint_dir = (
        Path(config.output_dir) / "checkpoints" / config.dataset_name / model_name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{config.file_prefix}{subject:02d}.pt"
    torch.save(
        {
            "model": best_state,
            "model_name": model_name,
            "subject": subject,
            "config": asdict(config),
            "validation_accuracy": best_validation,
            "test_metrics": test_metrics,
        },
        checkpoint,
    )
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "model": model_name,
        "dataset": config.dataset_name,
        "subject": subject,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_trials": int(data["train_y"].size),
        "validation_trials": int(data["validation_y"].size),
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "validation_accuracy": best_validation,
        **test_metrics,
        "excluded_train_artifacts": int(data["excluded_train_artifacts"]),
        "excluded_test_artifacts": int(data["excluded_test_artifacts"]),
        "duration_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "checkpoint_size_bytes": int(checkpoint.stat().st_size),
        "checkpoint": str(checkpoint),
        "training_config": asdict(config),
    }
    print(
        f"{model_name} {config.file_prefix}{subject:02d} test_acc={result['accuracy']:.4f} "
        f"kappa={result['kappa']:.4f}",
        flush=True,
    )
    del model, loaders, tensor_data, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["model"]), []).append(result)
    summary: dict[str, object] = {}
    for model, rows in grouped.items():
        summary[model] = {
            "mean_accuracy": float(np.mean([float(row["accuracy"]) for row in rows])),
            "std_accuracy": float(np.std([float(row["accuracy"]) for row in rows])),
            "mean_macro_f1": float(np.mean([float(row["macro_f1"]) for row in rows])),
            "mean_kappa": float(np.mean([float(row["kappa"]) for row in rows])),
            "subjects": len(rows),
        }
    return summary


def save_results(results: list[dict[str, object]], config: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs_by_model = {
        str(row["model"]): row.get("training_config", asdict(config)) for row in results
    }
    payload = {
        "configs_by_model": configs_by_model,
        "results": results,
        "summary": summarize_results(results),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

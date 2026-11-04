"""Leakage-controlled public pretraining and participant calibration for SE-MHAF."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import binom
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .recording import load_mi_recordings
from .semhaf import (
    CLASSES, MONTAGE_ORDER_SOURCE, MONTAGE_SOURCE, SEMHAFBundle,
    filter_epochs, fit_normalization, load_semhaf_bundle, new_bundle,
    public_channel_indices, validate_epochs,
)
from .source import DEFAULT_CHANNEL_NAMES


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    patience: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    cpu_threads: int = 4


def seed_training(config: TrainingConfig) -> None:
    if min(config.epochs, config.patience, config.batch_size, config.cpu_threads) < 1 or config.learning_rate <= 0:
        raise ValueError("Training epochs, patience, batch size, threads and learning rate must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _require_classes(labels: np.ndarray, minimum: int, split: str) -> None:
    counts = {name: int(np.sum(labels == index)) for index, name in enumerate(CLASSES)}
    if not np.isin(labels, (0, 1)).all() or min(counts.values()) < minimum:
        raise ValueError(f"{split} needs at least {minimum} clean trials per class; counts={counts}")


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    threshold = float(binom.ppf(0.95, len(labels), 0.5) / len(labels))
    accuracy = float(accuracy_score(labels, predictions))
    balanced = float(balanced_accuracy_score(labels, predictions))
    return {
        "n_trials": int(len(labels)), "class_counts": {name: int(np.sum(labels == index)) for index, name in enumerate(CLASSES)},
        "accuracy": accuracy, "balanced_accuracy": balanced,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "chance_accuracy_95_percent": threshold,
        "above_chance_95_percent": bool(accuracy > threshold and balanced > 0.5),
    }


def _predict_filtered(bundle: SEMHAFBundle, x: np.ndarray, batch_size: int) -> np.ndarray:
    bundle.model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            values = (x[start:start + batch_size] - bundle.mean) / bundle.std
            logits = bundle.model(torch.from_numpy(values).to(bundle.device))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def evaluate_epochs(bundle: SEMHAFBundle, epochs: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> dict:
    _require_classes(labels, 1, "evaluation")
    # Filter in batches to bound memory when evaluating a complete public session.
    probabilities = []
    for start in range(0, len(epochs), batch_size):
        probabilities.append(bundle.predict_proba(epochs[start:start + batch_size], 250, DEFAULT_CHANNEL_NAMES))
    return classification_metrics(labels, np.concatenate(probabilities))


def _selection_score(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray | None) -> float:
    if groups is None:
        return float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))
    return float(np.mean([
        balanced_accuracy_score(labels[groups == group], probabilities[groups == group].argmax(axis=1))
        for group in np.unique(groups)
    ]))


def fit_model(
    bundle: SEMHAFBundle, train_epochs: np.ndarray, train_labels: np.ndarray,
    validation_epochs: np.ndarray, validation_labels: np.ndarray, config: TrainingConfig,
    *, validation_groups: np.ndarray | None = None,
) -> dict:
    """Only train epochs fit normalization/gradients; validation selects the epoch."""
    seed_training(config)
    _require_classes(train_labels, 3, "training")
    _require_classes(validation_labels, 2, "validation")
    train_x, validation_x = filter_epochs(train_epochs), filter_epochs(validation_epochs)
    bundle.mean, bundle.std = fit_normalization(train_x)
    train_x -= bundle.mean
    train_x /= bundle.std
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_labels.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config.seed))
    # All parameters, including the new spatial stem and binary heads, are trainable.
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_score, best_loss, best_epoch = -1.0, float("inf"), 0
    best_state, history = None, []
    for epoch in range(config.epochs + 1):
        train_loss = None
        if epoch:
            bundle.model.train()
            total_loss = 0.0
            for batch, labels in loader:
                batch, labels = batch.to(bundle.device), labels.to(bundle.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(bundle.model(batch), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            train_loss = total_loss / len(dataset)
        probabilities = _predict_filtered(bundle, validation_x, config.batch_size)
        score = _selection_score(validation_labels, probabilities, validation_groups)
        loss_value = float(-np.log(np.clip(probabilities[np.arange(len(validation_labels)), validation_labels], 1e-8, 1.0)).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_balanced_accuracy": score, "validation_loss": loss_value})
        if score > best_score or (score == best_score and loss_value < best_loss):
            best_score, best_loss, best_epoch = score, loss_value, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in bundle.model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
        if epoch - best_epoch >= config.patience:
            break
    bundle.model.load_state_dict(best_state, strict=True)
    bundle.model.eval()
    return {
        "config": asdict(config), "best_epoch": best_epoch, "epochs_completed": history[-1]["epoch"],
        "best_validation_selection_score": best_score,
        "selection_metric": "mean_subject_balanced_accuracy" if validation_groups is not None else "balanced_accuracy",
        "normalization_fit": "training_trials_only", "trainable_parameters": "all",
        "history": history, "validation": classification_metrics(validation_labels, _predict_filtered(bundle, validation_x, config.batch_size)),
    }


def load_public_trials(processed_dir: str | Path, subjects: Sequence[int], session: str = "T") -> dict:
    if session not in {"T", "E"} or not subjects or len(set(subjects)) != len(subjects) or any(subject not in range(1, 10) for subject in subjects):
        raise ValueError("Use unique BCIC2a subjects 1..9 and session T or E")
    root = Path(processed_dir)
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", {})
        if (config.get("sfreq"), config.get("epoch_start"), config.get("epoch_end")) != (250.0, 2.0, 6.0):
            raise ValueError("Public data must use trial onset +2..+6 seconds at 250 Hz")
    epochs, labels, groups, files, details = [], [], [], [], []
    for subject in subjects:
        path = root / f"A{subject:02d}{session}.npz"
        with np.load(path, allow_pickle=False) as loaded:
            indices = public_channel_indices(loaded["channel_names"].tolist())
            raw, targets = loaded["raw_x"], np.asarray(loaded["y"], dtype=np.int64)
            artifacts = np.asarray(loaded["artifact"], dtype=bool)
            if raw.ndim != 3 or raw.shape[1:] != (22, 1000) or len(raw) != len(targets) or len(raw) != len(artifacts):
                raise ValueError(f"Invalid public trial arrays in {path}")
            if float(loaded["sfreq"]) != 250.0:
                raise ValueError(f"Public sample rate must be 250 Hz: {path}")
            keep = np.isin(targets, (0, 1)) & ~artifacts
            selected = validate_epochs(raw[keep][:, indices], 250, DEFAULT_CHANNEL_NAMES)
            _require_classes(targets[keep], 2, str(path))
            epochs.append(selected)
            labels.append(targets[keep])
            groups.append(np.full(int(keep.sum()), subject, dtype=np.int64))
            files.append(str(path.resolve()))
            details.append({"subject": subject, "session": session, "clean_left_right_trials": int(keep.sum()), "source_channel_indices_zero_based": indices})
    return {"epochs": np.concatenate(epochs), "labels": np.concatenate(labels), "groups": np.concatenate(groups), "files": files, "sessions": details}


def pretrain_public(
    processed_dir: str | Path, train_subjects: Sequence[int], validation_subjects: Sequence[int],
    config: TrainingConfig, *, device: str = "auto",
) -> SEMHAFBundle:
    if set(train_subjects) & set(validation_subjects) or len(train_subjects) < 2 or not validation_subjects:
        raise ValueError("Pretraining requires >=2 train subjects and disjoint validation subjects")
    seed_training(config)
    train = load_public_trials(processed_dir, train_subjects, "T")
    validation = load_public_trials(processed_dir, validation_subjects, "T")
    bundle = new_bundle(device=device, metadata={
        "provenance": {"data_origin": "public", "dataset": "BCIC2a", "calibration_passed": False, "human_mi_verified": False},
        "training_files": train["files"], "validation_files": validation["files"],
        "train_subjects": list(train_subjects), "validation_subjects": list(validation_subjects),
        "montage_source": MONTAGE_SOURCE, "montage_order_source": MONTAGE_ORDER_SOURCE,
        "initialization": "random_seeded",
        "initialization_reason": "Fresh 16-channel binary CAR training on T sessions; archived subject-specific checkpoints have incompatible montages and prior validation histories.",
        "public_epoch": "trial_onset_plus_2_to_6_seconds", "raw_units": "uV",
    })
    report = fit_model(bundle, train["epochs"], train["labels"], validation["epochs"], validation["labels"], config, validation_groups=validation["groups"])
    report.update({"stage": "public_pretraining", "data_origin": "public", "train_sessions": train["sessions"],
                   "validation_sessions": validation["sessions"], "split": "disjoint_subject_T_validation",
                   "E_sessions_used_during_training": False, "final_E_evaluation": "pending_separate_evaluate_only_command"})
    bundle.training_report = report
    return bundle


def evaluate_public(bundle: SEMHAFBundle, processed_dir: str | Path, subjects: Sequence[int], batch_size: int = 32) -> dict:
    seen = set(bundle.metadata.get("train_subjects", []))
    validation_subjects = set(bundle.metadata.get("validation_subjects", []))
    records = []
    for subject in subjects:
        data = load_public_trials(processed_dir, [subject], "E")
        category = "cross_session_seen_subject" if subject in seen else (
            "heldout_subject_used_for_T_validation" if subject in validation_subjects else "unseen_subject")
        records.append({"subject": subject, "session": "E", "category": category,
                        **evaluate_epochs(bundle, data["epochs"], data["labels"], batch_size)})
    categories = sorted({record["category"] for record in records})
    return {"stage": "public_final_E_evaluation", "data_origin": "public", "model_selection": False, "subjects": records,
            "category_means": {category: {
                "subject_count": sum(record["category"] == category for record in records),
                "mean_subject_accuracy": float(np.mean([record["accuracy"] for record in records if record["category"] == category])),
                "mean_subject_balanced_accuracy": float(np.mean([record["balanced_accuracy"] for record in records if record["category"] == category])),
            } for category in categories}}


def load_personal_trials(paths: Sequence[str | Path], subject_id: str) -> dict:
    canonical = [str(Path(path).resolve()) for path in paths]
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("Provide unique recording paths")
    epochs, labels, groups, sessions, metadata, origins = [], [], [], [], [], set()
    session_ids = set()
    for file_index, path in enumerate(canonical):
        with np.load(path, allow_pickle=False) as header:
            if float(np.asarray(header["sample_rate"]).item()) != 250.0:
                raise ValueError(f"Recording {path} must use exactly 250 Hz")
        data = load_mi_recordings([path])
        meta = data["metadata"][0]
        if str(meta.get("subject_id")) != subject_id:
            raise ValueError(f"Recording {path} does not belong to subject {subject_id}")
        session = str(meta.get("session_id", ""))
        if not session or session in session_ids:
            raise ValueError("Recordings require unique non-empty session IDs")
        session_ids.add(session)
        if meta.get("units", "uV") != "uV":
            raise ValueError(f"Recording {path} must contain microvolts (uV)")
        provenance = meta.get("provenance", {})
        kind = provenance.get("kind")
        if not kind:
            kind = {"brainflow": "hardware", "lsl": "lsl", "synthetic": "synthetic", "synthetic_mi": "synthetic"}.get(meta.get("configuration", {}).get("backend"))
        origin = {"hardware": "hardware_recorded", "lsl": "lsl_recorded", "synthetic": "synthetic"}.get(kind)
        if origin is None:
            raise ValueError(f"Recording {path} has unsupported or unverified source provenance: {kind!r}")
        backend = provenance.get("backend") or meta.get("configuration", {}).get("backend")
        if kind in {"hardware", "lsl"} and backend in {"synthetic", "synthetic_mi", "replay"}:
            raise ValueError(f"Recording {path} has conflicting source provenance")
        if kind in {"hardware", "lsl"} and provenance.get("label_source") == "scripted_simulation":
            raise ValueError(f"Recording {path} has simulated labels marked as hardware acquisition")
        origins.add(origin)
        targets = np.asarray(data["labels"]).astype(str)
        if not np.isin(targets, CLASSES).all():
            raise ValueError("Personal recordings must contain only left_hand and right_hand labels")
        keep = ~np.asarray(data["artifact"], dtype=bool)
        values = validate_epochs(data["epochs"][keep], int(data["sample_rate"]), data["channel_names"].tolist())
        epochs.append(values)
        labels.append(np.asarray([CLASSES.index(label) for label in targets[keep]], dtype=np.int64))
        groups.extend([f"{file_index}:{int(block)}" for block in data["blocks"][keep]])
        sessions.extend([session] * int(keep.sum()))
        metadata.append({"file": path, "metadata": meta, "clean_trials": int(keep.sum()), "artifact_trials": int((~keep).sum())})
    if len(origins) != 1:
        raise ValueError("Personal training requires a single data origin; keep synthetic and hardware sources separate")
    return {"epochs": np.concatenate(epochs), "labels": np.concatenate(labels), "groups": np.asarray(groups),
            "sessions": np.asarray(sessions), "recordings": metadata, "files": canonical, "data_origin": origins.pop()}


def personal_splits(labels: np.ndarray, groups: np.ndarray, sessions: np.ndarray, *, seed: int = 42, holdout_session: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Later group/session is final evaluation; an earlier split selects epochs."""
    ordered_groups = list(dict.fromkeys(groups.tolist()))
    if holdout_session is not None:
        if holdout_session not in sessions:
            raise ValueError(f"Unknown holdout session {holdout_session!r}")
        test = np.flatnonzero(sessions == holdout_session)
        development = np.flatnonzero(sessions != holdout_session)
        test_description = f"independent_session:{holdout_session}"
    else:
        if len(ordered_groups) < 2:
            raise ValueError("Personal calibration needs at least two blocks or an independent holdout session")
        test = np.flatnonzero(groups == ordered_groups[-1])
        development = np.flatnonzero(groups != ordered_groups[-1])
        test_description = f"last_block_in_input_order:{ordered_groups[-1]}"
    development_groups = list(dict.fromkeys(groups[development].tolist()))
    if len(development_groups) >= 2:
        validation = development[groups[development] == development_groups[-1]]
        train = development[groups[development] != development_groups[-1]]
        validation_description = f"last_development_block:{development_groups[-1]}"
    else:
        _require_classes(labels[development], 5, "development before internal split")
        train, validation = train_test_split(development, test_size=max(4, int(np.ceil(len(development) * 0.2))), stratify=labels[development], random_state=seed)
        validation_description = "stratified_20_percent_within_development_block"
    _require_classes(labels[train], 3, "training")
    _require_classes(labels[validation], 2, "early-stopping validation")
    _require_classes(labels[test], 2, "independent final holdout")
    return train, validation, test, {
        "final_holdout": test_description, "early_stopping_validation": validation_description,
        "train_indices": train.tolist(), "validation_indices": validation.tolist(), "final_holdout_indices": test.tolist(),
        "trial_counts": {"train": int(len(train)), "validation": int(len(validation)), "final_holdout": int(len(test))},
        "class_counts": {name: {label: int(np.sum(labels[indices] == index)) for index, label in enumerate(CLASSES)}
                         for name, indices in (("train", train), ("validation", validation), ("final_holdout", test))},
        "small_sample": bool(min(np.sum(labels[train] == value) for value in (0, 1)) < 10),
        "final_holdout_used_for_training_or_selection": False,
    }


def finetune_personal(
    pretrained: str | Path, paths: Sequence[str | Path], subject_id: str, config: TrainingConfig,
    *, device: str = "auto", holdout_session: str | None = None,
) -> SEMHAFBundle:
    seed_training(config)
    bundle = load_semhaf_bundle(pretrained, device=device)
    data = load_personal_trials(paths, subject_id)
    if bundle.subject_id not in {"public", subject_id}:
        raise ValueError(f"Checkpoint belongs to another participant: {bundle.subject_id}")
    if bundle.provenance.get("data_origin") != "public":
        raise ValueError("Start this evaluation from the public pretrained checkpoint to keep personal holdout history independent")
    train, validation, test, split = personal_splits(data["labels"], data["groups"], data["sessions"], seed=config.seed, holdout_session=holdout_session)
    before = evaluate_epochs(bundle, data["epochs"][test], data["labels"][test], config.batch_size)
    report = fit_model(bundle, data["epochs"][train], data["labels"][train], data["epochs"][validation], data["labels"][validation], config)
    after = evaluate_epochs(bundle, data["epochs"][test], data["labels"][test], config.batch_size)
    human_verified = data["data_origin"] != "synthetic" and all(record["metadata"].get("provenance", {}).get("human_mi_verified") is True for record in data["recordings"])
    parent_metadata = copy.deepcopy(bundle.metadata)
    bundle.subject_id = subject_id
    bundle.metadata = {
        "provenance": {"data_origin": data["data_origin"], "human_mi_verified": human_verified,
                       "calibration_passed": bool(data["data_origin"] != "synthetic" and after["above_chance_95_percent"])},
        "training_files": data["files"], "pretrained_checkpoint": str(Path(pretrained).resolve()),
        "public_pretraining": parent_metadata, "recordings": data["recordings"],
    }
    report.update({"stage": "personal_finetuning", "subject_id": subject_id, "data_origin": data["data_origin"],
                   "split": split, "before_finetuning": before, "after_finetuning": after,
                   "calibration_passed": bundle.provenance["calibration_passed"],
                   "human_mi_verified": human_verified, "final_holdout_refit": False})
    bundle.training_report = report
    return bundle

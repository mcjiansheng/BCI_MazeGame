"""Subject-specific CSP+LDA baseline for LK-Mini motor imagery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import binom
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class MIModelConfig:
    sample_rate: int = 250
    low_freq: float = 7.0
    high_freq: float = 30.0
    filter_order: int = 4
    csp_components: int = 4
    epoch_seconds: float = 4.0


class EpochBandpass(BaseEstimator, TransformerMixin):
    """Apply the same window-local band-pass in training and online prediction."""

    def __init__(self, sample_rate: int = 250, low_freq: float = 7.0, high_freq: float = 30.0, order: int = 4):
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order

    def fit(self, x: np.ndarray, y: np.ndarray | None = None):
        _validate_epochs(x)
        if not 0 < self.low_freq < self.high_freq < self.sample_rate / 2:
            raise ValueError("Invalid band-pass frequencies")
        self.sos_ = butter(
            self.order,
            (self.low_freq, self.high_freq),
            btype="bandpass",
            fs=self.sample_rate,
            output="sos",
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        epochs = _validate_epochs(x)
        if not hasattr(self, "sos_"):
            raise RuntimeError("EpochBandpass is not fitted")
        filtered = sosfiltfilt(self.sos_, epochs, axis=-1)
        return np.nan_to_num(filtered, copy=False).astype(np.float64, copy=False)


class BinaryCSP(BaseEstimator, TransformerMixin):
    """Regularized binary Common Spatial Pattern transformer."""

    def __init__(self, n_components: int = 4, regularization: float = 0.1):
        self.n_components = n_components
        self.regularization = regularization

    @staticmethod
    def _normalized_covariance(epoch: np.ndarray) -> np.ndarray:
        covariance = epoch @ epoch.T
        trace = float(np.trace(covariance))
        return covariance / max(trace, 1e-12)

    def fit(self, x: np.ndarray, y: np.ndarray):
        epochs = _validate_epochs(x)
        labels = np.asarray(y)
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("BinaryCSP requires exactly two classes")
        n_channels = epochs.shape[1]
        n_components = min(int(self.n_components), n_channels)
        if n_components < 2:
            raise ValueError("At least two CSP components are required")
        class_covariances = []
        identity = np.eye(n_channels) / n_channels
        for class_name in classes:
            covariance = np.mean(
                [self._normalized_covariance(epoch) for epoch in epochs[labels == class_name]],
                axis=0,
            )
            covariance = (1.0 - self.regularization) * covariance + self.regularization * identity
            class_covariances.append(covariance)
        eigenvalues, eigenvectors = eigh(
            class_covariances[0], class_covariances[0] + class_covariances[1]
        )
        order = np.argsort(eigenvalues)
        low_count = n_components // 2
        high_count = n_components - low_count
        selected = np.r_[order[:low_count], order[-high_count:]]
        self.filters_ = eigenvectors[:, selected].T
        self.classes_ = classes
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        epochs = _validate_epochs(x)
        if not hasattr(self, "filters_"):
            raise RuntimeError("BinaryCSP is not fitted")
        projected = np.einsum("kc,nct->nkt", self.filters_, epochs)
        variances = np.var(projected, axis=-1)
        normalized = variances / np.maximum(variances.sum(axis=1, keepdims=True), 1e-12)
        return np.log(np.maximum(normalized, 1e-12))


def _validate_epochs(x: np.ndarray) -> np.ndarray:
    epochs = np.asarray(x, dtype=np.float64)
    if epochs.ndim != 3:
        raise ValueError("epochs must be trials x channels x samples")
    if epochs.shape[0] == 0 or epochs.shape[1] < 2 or epochs.shape[2] < 32:
        raise ValueError(f"Invalid epoch shape: {epochs.shape}")
    return epochs


def build_csp_lda_pipeline(config: MIModelConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "bandpass",
                EpochBandpass(
                    config.sample_rate,
                    config.low_freq,
                    config.high_freq,
                    config.filter_order,
                ),
            ),
            ("csp", BinaryCSP(config.csp_components, regularization=0.1)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


@dataclass
class MIModelBundle:
    pipeline: Pipeline
    config: MIModelConfig
    channel_names: tuple[str, ...]
    classes: tuple[str, ...]
    subject_id: str
    training_files: tuple[str, ...]

    def validate_input(self, epoch: np.ndarray, sample_rate: int, channel_names: Sequence[str]) -> np.ndarray:
        values = np.asarray(epoch, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, :, :]
        expected_samples = int(round(self.config.epoch_seconds * self.config.sample_rate))
        if sample_rate != self.config.sample_rate:
            raise ValueError(f"Model requires {self.config.sample_rate} Hz, got {sample_rate} Hz")
        if tuple(channel_names) != self.channel_names:
            raise ValueError(
                f"Model channel order {self.channel_names} does not match input {tuple(channel_names)}"
            )
        if values.shape[1:] != (len(self.channel_names), expected_samples):
            raise ValueError(
                f"Model requires (*, {len(self.channel_names)}, {expected_samples}), got {values.shape}"
            )
        return values

    def predict_proba(self, epoch: np.ndarray, sample_rate: int, channel_names: Sequence[str]) -> np.ndarray:
        values = self.validate_input(epoch, sample_rate, channel_names)
        return self.pipeline.predict_proba(values)


def _cross_validation_splits(labels: np.ndarray, groups: np.ndarray, seed: int):
    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=min(5, unique_groups.size), shuffle=True, random_state=seed
        )
        return list(splitter.split(np.zeros(len(labels)), labels, groups)), "stratified_group"
    smallest_class = min(np.sum(labels == value) for value in np.unique(labels))
    if smallest_class < 2:
        raise ValueError("Each class needs at least two clean trials")
    splitter = StratifiedKFold(
        n_splits=min(5, int(smallest_class)), shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(labels)), labels)), "stratified"


def _band_effect_sizes(
    epochs: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, str],
    sample_rate: int,
    low: float,
    high: float,
) -> list[float]:
    frequencies, spectrum = welch(
        epochs,
        fs=sample_rate,
        axis=-1,
        nperseg=min(epochs.shape[-1], sample_rate * 2),
    )
    mask = (frequencies >= low) & (frequencies <= high)
    log_power = np.log(np.maximum(spectrum[..., mask].mean(axis=-1), 1e-12))
    first, second = log_power[labels == classes[0]], log_power[labels == classes[1]]
    difference = first.mean(axis=0) - second.mean(axis=0)
    pooled = np.sqrt((first.var(axis=0, ddof=1) + second.var(axis=0, ddof=1)) / 2.0)
    return (difference / np.maximum(pooled, 1e-12)).astype(float).tolist()


def train_subject_model(
    epochs: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[int],
    *,
    channel_names: Sequence[str],
    subject_id: str,
    training_files: Sequence[str],
    config: MIModelConfig,
    seed: int = 42,
) -> tuple[MIModelBundle, dict[str, object]]:
    values = _validate_epochs(epochs)
    targets = np.asarray(labels).astype(str)
    group_values = np.asarray(groups, dtype=np.int64)
    if len(targets) != len(values) or len(group_values) != len(values):
        raise ValueError("epochs, labels and groups must have equal length")
    classes = tuple(np.unique(targets).tolist())
    if len(classes) != 2:
        raise ValueError(
            f"The first validated LK-Mini baseline is binary; got classes {classes}. "
            "Collect left_hand and right_hand first."
        )
    class_counts = {class_name: int(np.sum(targets == class_name)) for class_name in classes}
    if min(class_counts.values()) < 2:
        raise ValueError("Each class needs at least two clean trials")
    expected_samples = int(round(config.epoch_seconds * config.sample_rate))
    if values.shape[1:] != (len(channel_names), expected_samples):
        raise ValueError(
            f"Expected epochs (*, {len(channel_names)}, {expected_samples}), got {values.shape}"
        )

    splits, validation_kind = _cross_validation_splits(targets, group_values, seed)
    predictions = np.empty(targets.shape, dtype=targets.dtype)
    probabilities = np.zeros((len(targets), len(classes)), dtype=np.float64)
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(splits, start=1):
        pipeline = build_csp_lda_pipeline(config)
        pipeline.fit(values[train_indices], targets[train_indices])
        fold_predictions = pipeline.predict(values[test_indices])
        fold_probabilities = pipeline.predict_proba(values[test_indices])
        predictions[test_indices] = fold_predictions
        for source_column, class_name in enumerate(pipeline.classes_):
            probabilities[test_indices, classes.index(str(class_name))] = fold_probabilities[:, source_column]
        folds.append(
            {
                "fold": fold_index,
                "train_trials": int(len(train_indices)),
                "validation_trials": int(len(test_indices)),
                "accuracy": float(accuracy_score(targets[test_indices], fold_predictions)),
            }
        )

    temporal_holdout = None
    unique_groups = np.unique(group_values)
    if unique_groups.size >= 2:
        holdout_group = unique_groups[-1]
        holdout_indices = np.flatnonzero(group_values == holdout_group)
        development_indices = np.flatnonzero(group_values != holdout_group)
        development_classes = np.unique(targets[development_indices])
        if development_classes.size == 2 and holdout_indices.size:
            holdout_pipeline = build_csp_lda_pipeline(config)
            holdout_pipeline.fit(values[development_indices], targets[development_indices])
            holdout_predictions = holdout_pipeline.predict(values[holdout_indices])
            holdout_accuracy = float(
                accuracy_score(targets[holdout_indices], holdout_predictions)
            )
            holdout_chance_95 = float(
                binom.ppf(0.95, len(holdout_indices), 1.0 / len(classes))
                / len(holdout_indices)
            )
            temporal_holdout = {
                "group": int(holdout_group),
                "development_trials": int(len(development_indices)),
                "validation_trials": int(len(holdout_indices)),
                "accuracy": holdout_accuracy,
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        targets[holdout_indices], holdout_predictions
                    )
                ),
                "chance_accuracy_95_percent": holdout_chance_95,
                "above_chance_95_percent": bool(holdout_accuracy > holdout_chance_95),
            }

    pipeline = build_csp_lda_pipeline(config)
    pipeline.fit(values, targets)
    bundle = MIModelBundle(
        pipeline=pipeline,
        config=config,
        channel_names=tuple(channel_names),
        classes=tuple(str(value) for value in pipeline.classes_),
        subject_id=subject_id,
        training_files=tuple(training_files),
    )
    chance_95 = float(binom.ppf(0.95, len(targets), 1.0 / len(classes)) / len(targets))
    cross_validated_accuracy = float(accuracy_score(targets, predictions))
    cross_validated_above_chance = cross_validated_accuracy > chance_95
    temporal_above_chance = bool(
        temporal_holdout is None or temporal_holdout["above_chance_95_percent"]
    )
    report = {
        "subject_id": subject_id,
        "classes": list(classes),
        "channel_names": list(channel_names),
        "config": asdict(config),
        "validation": validation_kind,
        "n_trials": int(len(targets)),
        "class_counts": class_counts,
        "accuracy": cross_validated_accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "kappa": float(cohen_kappa_score(targets, predictions)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=classes).tolist(),
        "chance_accuracy_95_percent": chance_95,
        "cross_validated_above_chance_95_percent": bool(cross_validated_above_chance),
        "temporal_holdout": temporal_holdout,
        "above_chance_95_percent": bool(
            cross_validated_above_chance and temporal_above_chance
        ),
        "physiology": {
            "effect_size_definition": f"Cohen d of log band power: {classes[0]} minus {classes[1]}",
            "mu_8_13_hz_by_channel": _band_effect_sizes(
                values, targets, classes, config.sample_rate, 8.0, 13.0
            ),
            "beta_13_30_hz_by_channel": _band_effect_sizes(
                values, targets, classes, config.sample_rate, 13.0, 30.0
            ),
            "interpretation": (
                "Effect sizes describe class-related sensorimotor rhythm differences. "
                "They support, but do not alone prove, genuine motor imagery."
            ),
        },
        "folds": folds,
    }
    return bundle, report


def save_model_bundle(bundle: MIModelBundle, report: dict[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def load_model_bundle(path: str | Path) -> MIModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, MIModelBundle):
        raise TypeError("File does not contain an MIModelBundle")
    return bundle

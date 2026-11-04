"""Group-held-out CSP + LDA with within-group label permutation controls."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from bci_maze.lk_mini.model import MIModelConfig, build_csp_lda_pipeline


CLASSES = ("left_hand", "right_hand")


def group_splits(labels: np.ndarray, sessions: np.ndarray, blocks: np.ndarray, split: str = "auto"):
    """Keep each session/block together, with no random-trial fallback."""
    if split not in {"auto", "session", "block"}:
        raise ValueError("split 必须为 auto、session 或 block。")
    labels, sessions, blocks = np.asarray(labels), np.asarray(sessions).astype(str), np.asarray(blocks).astype(str)
    if labels.ndim != 1 or sessions.shape != labels.shape or blocks.shape != labels.shape:
        raise ValueError("每个试次须有一个类别、会话和分组标签。")
    if set(labels.tolist()) != set(CLASSES):
        raise ValueError("分组评估需要左手和右手两类有效试次。")
    method = "session" if split == "session" or (split == "auto" and np.unique(sessions).size >= 2) else "block"
    # Structured pairs avoid collisions in user-supplied session names.
    groups = sessions if method == "session" else np.asarray([
        f"{len(session)}:{session}/block:{block}" for session, block in zip(sessions, blocks)
    ])
    if np.unique(groups).size < 2:
        raise ValueError(f"按{('会话' if method == 'session' else '实验组')}划分需要至少两个独立组，当前为 {np.unique(groups).size} 组。")
    folds = list(LeaveOneGroupOut().split(np.zeros(len(labels)), labels, groups))
    for index, (train, test) in enumerate(folds, 1):
        if set(labels[train]) != set(CLASSES) or set(labels[test]) != set(CLASSES):
            counts = lambda selected: "/".join(str(int(np.count_nonzero(labels[selected] == label))) for label in CLASSES)
            raise ValueError(f"第 {index} 折须在训练集和测试集各包含两类；当前训练左/右为 {counts(train)}，测试左/右为 {counts(test)}。")
        if any(np.count_nonzero(labels[train] == label) < 2 for label in CLASSES):
            raise ValueError(f"第 {index} 折训练集每类需要至少两个有效试次。")
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("训练组与测试组重叠。")
    return groups, folds, "leave_one_session_out" if method == "session" else "leave_one_block_out"


def grouped_benchmark(epochs: np.ndarray, labels: np.ndarray, sessions: np.ndarray,
                      blocks: np.ndarray, sample_rate: int, *, permutations: int = 30,
                      seed: int = 42, split: str = "auto") -> dict:
    if type(permutations) is not int or not 0 <= permutations <= 1000:
        raise ValueError("置乱次数须为 0–1000 的整数。")
    epochs, labels = np.asarray(epochs, dtype=np.float64), np.asarray(labels).astype(str)
    groups, folds, method = group_splits(labels, sessions, blocks, split)
    if epochs.ndim != 3 or len(epochs) != len(labels) or epochs.shape[1] < 2 or not np.isfinite(epochs).all():
        raise ValueError("CSP 输入须为有限值的试次 × EEG 通道 × 采样点数组，至少两个通道。")
    config = MIModelConfig(sample_rate=sample_rate, low_freq=7.0, high_freq=30.0,
                           csp_components=min(4, epochs.shape[1]), epoch_seconds=epochs.shape[-1] / sample_rate)

    def cross_validate(target: np.ndarray, details: bool = False):
        predicted = np.empty(target.shape, dtype=target.dtype)
        fold_results = []
        for index, (train, test) in enumerate(folds, 1):
            # A fresh pipeline fits temporal filtering, CSP and LDA inside this fold.
            pipeline = build_csp_lda_pipeline(config)
            pipeline.fit(epochs[train], target[train])
            predicted[test] = pipeline.predict(epochs[test])
            if details:
                fold_results.append(dict(
                    fold=index, train_groups=np.unique(groups[train]).tolist(),
                    test_groups=np.unique(groups[test]).tolist(),
                    train_indices=train.tolist(), test_indices=test.tolist(),
                    n_train=int(train.size), n_test=int(test.size),
                    train_class_counts={label: int(np.count_nonzero(target[train] == label)) for label in CLASSES},
                    test_class_counts={label: int(np.count_nonzero(target[test] == label)) for label in CLASSES},
                    accuracy=float(accuracy_score(target[test], predicted[test])),
                    balanced_accuracy=float(balanced_accuracy_score(target[test], predicted[test])),
                ))
        return predicted, fold_results

    prediction, details = cross_validate(labels, details=True)
    observed = float(balanced_accuracy_score(labels, prediction))
    rng = np.random.default_rng(seed)
    null_scores = []
    # Preserve both group membership and class counts in each group on every draw.
    group_members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for _ in range(permutations):
        shuffled = labels.copy()
        for members in group_members:
            shuffled[members] = rng.permutation(labels[members])
        permuted_predictions, _ = cross_validate(shuffled)
        null_scores.append(float(balanced_accuracy_score(shuffled, permuted_predictions)))
    return dict(
        status="completed", split_method=method, n_groups=int(np.unique(groups).size), n_splits=len(folds),
        accuracy=float(accuracy_score(labels, prediction)), balanced_accuracy=observed,
        chance_balanced_accuracy=0.5, class_order=list(CLASSES),
        confusion_matrix=confusion_matrix(labels, prediction, labels=CLASSES).tolist(),
        folds=details, predictions=prediction.tolist(), true_labels=labels.tolist(),
        features="EEG 7–30 Hz 带通、CSP 对数归一化方差、收缩 LDA",
        permutation=dict(n=permutations, observed_score=observed, null_scores=null_scores,
                         null_mean=float(np.mean(null_scores)) if null_scores else None,
                         p_value=(1 + sum(score >= observed for score in null_scores)) / (permutations + 1) if null_scores else None,
                         seed=seed, scheme="within_group_labels", score="pooled_out_of_fold_balanced_accuracy",
                         formula="(1 + count(null_score >= observed_score)) / (N + 1)"),
    )

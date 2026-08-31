import numpy as np
import pytest

from bci_maze.lk_mini.model import (
    MIModelConfig,
    load_model_bundle,
    save_model_bundle,
    train_subject_model,
)


def make_mi_epochs():
    rng = np.random.default_rng(42)
    sample_rate = 250
    time = np.arange(1000) / sample_rate
    epochs, labels, groups = [], [], []
    for block in range(4):
        for label in ("left_hand", "right_hand"):
            for _ in range(10):
                epoch = rng.normal(0, 2.0, (8, 1000))
                if label == "left_hand":
                    epoch[0] += 14 * np.sin(2 * np.pi * 10 * time)
                    epoch[2] += 10 * np.sin(2 * np.pi * 12 * time)
                else:
                    epoch[1] += 14 * np.sin(2 * np.pi * 10 * time)
                    epoch[4] += 10 * np.sin(2 * np.pi * 12 * time)
                epochs.append(epoch)
                labels.append(label)
                groups.append(block)
    return np.asarray(epochs), np.asarray(labels), np.asarray(groups)


def test_csp_lda_training_round_trip(tmp_path):
    epochs, labels, groups = make_mi_epochs()
    channels = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4")
    bundle, report = train_subject_model(
        epochs,
        labels,
        groups,
        channel_names=channels,
        subject_id="TEST",
        training_files=("synthetic.npz",),
        config=MIModelConfig(),
    )
    assert report["accuracy"] > 0.9
    assert report["temporal_holdout"]["accuracy"] > 0.9
    assert report["above_chance_95_percent"]
    path = save_model_bundle(bundle, report, tmp_path / "model.joblib")
    loaded = load_model_bundle(path)
    probabilities = loaded.predict_proba(epochs[0], 250, channels)
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)
    with pytest.raises(ValueError):
        loaded.predict_proba(epochs[0], 500, channels)

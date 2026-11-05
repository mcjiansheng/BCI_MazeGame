import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

from bci_maze.lk_mini.model import (
    BinaryCSP,
    EpochBandpass,
    MIModelBundle,
    MIModelConfig,
    load_model_bundle,
    save_model_bundle,
    train_subject_model,
)
from bci_maze.lk_mini.source import DEFAULT_CHANNEL_NAMES, LEGACY_CHANNEL_NAMES
from bci_maze.lk_mini.online import ProbabilitySmoother, classify_mi_window


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
    bundle.metadata["provenance"] = {"data_origin": "synthetic"}
    path = save_model_bundle(bundle, report, tmp_path / "model.joblib")
    original_model = path.read_bytes()
    original_report = path.with_suffix(".json").read_bytes()
    with pytest.raises(TypeError, match="JSON serializable"):
        save_model_bundle(bundle, {"invalid": object()}, path)
    assert path.read_bytes() == original_model
    assert path.with_suffix(".json").read_bytes() == original_report
    loaded = load_model_bundle(path)
    probabilities = loaded.predict_proba(epochs[0], 250, channels)
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)
    assert loaded.provenance == {"data_origin": "synthetic"}
    with pytest.raises(ValueError):
        loaded.predict_proba(epochs[0], 500, channels)
    damaged = epochs[0].copy()
    damaged[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        loaded.predict_proba(damaged, 250, channels)

    # An old joblib bundle had no metadata and used the standalone CSP class.
    del loaded.metadata
    old_csp = BinaryCSP()
    old_csp.__dict__.update(loaded.pipeline.named_steps["csp"].__dict__)
    loaded.pipeline.set_params(csp=old_csp)
    legacy = tmp_path / "legacy.joblib"
    joblib.dump(loaded, legacy)
    loaded = load_model_bundle(legacy)
    assert not loaded.provenance
    np.testing.assert_array_equal(loaded.predict_proba(epochs[0], 250, channels), probabilities)


@pytest.mark.parametrize("estimator", [EpochBandpass(), BinaryCSP()])
def test_standalone_transformers_reject_nonfinite_input(estimator):
    rng = np.random.default_rng(42)
    epochs = rng.standard_normal((4, 2, 100))
    labels = np.array([0, 1, 0, 1])
    bad = epochs.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        estimator.fit(bad, labels)
    estimator.fit(epochs, labels)
    with pytest.raises(ValueError, match="NaN or infinity"):
        estimator.transform(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_bundle_checks_nonfinite_epochs_before_custom_pipeline(value):
    channels = ("C3", "C4")
    epochs = np.zeros((4, 2, 250))
    labels = ["left_hand", "right_hand"] * 2
    pipeline = make_pipeline(DummyClassifier()).fit(epochs, labels)
    bundle = MIModelBundle(pipeline, MIModelConfig(epoch_seconds=1), channels,
                           tuple(pipeline.classes_), "S01", ())
    np.testing.assert_array_equal(bundle.predict_proba(epochs[0], 250, channels), [[0.5, 0.5]])
    damaged = epochs[0].copy()
    damaged[0, 0] = value
    with pytest.raises(ValueError, match="NaN or infinity"):
        bundle.predict_proba(damaged, 250, channels)


def test_sixteen_channel_model_learns_upper_bank_and_rejects_mismatched_inputs(tmp_path):
    rng = np.random.default_rng(104)
    labels = np.tile(["left_hand", "right_hand"], 16)
    groups = np.repeat(np.arange(4), 8)
    epochs = rng.normal(0, 2, (32, 16, 1000))
    wave = np.sin(2 * np.pi * 10 * np.arange(1000) / 250)
    for index, label in enumerate(labels):
        epochs[index, 9 if label == "left_hand" else 15] += 15 * wave
    bundle, report = train_subject_model(
        epochs, labels, groups, channel_names=DEFAULT_CHANNEL_NAMES,
        subject_id="TEST16", training_files=(), config=MIModelConfig(),
    )
    assert report["accuracy"] > 0.9
    assert report["channel_count"] == 16
    assert len(report["physiology"]["mu_8_13_hz_by_channel"]) == 16
    assert bundle.pipeline.named_steps["csp"].filters_.shape == (4, 16)
    bundle = load_model_bundle(save_model_bundle(bundle, report, tmp_path / "mi16.joblib"))
    assert bundle.classes[np.argmax(bundle.predict_proba(epochs[0], 250, DEFAULT_CHANNEL_NAMES))] == labels[0]
    with pytest.raises(ValueError, match="channel order"):
        bundle.predict_proba(epochs[0, :8], 250, LEGACY_CHANNEL_NAMES)
    wrong_names = list(DEFAULT_CHANNEL_NAMES)
    wrong_names[8], wrong_names[15] = wrong_names[15], wrong_names[8]
    with pytest.raises(ValueError, match="channel order"):
        bundle.predict_proba(epochs[0], 250, wrong_names)
    damaged = epochs[0].copy()
    damaged[15] = 0
    result = classify_mi_window(
        bundle, ProbabilitySmoother(bundle.classes, history=1),
        damaged, 250, DEFAULT_CHANNEL_NAMES,
    )
    assert result.bad_channels == ("CP2",)
    assert result.decision.reason == "bad_signal"
    assert not result.decision.accepted

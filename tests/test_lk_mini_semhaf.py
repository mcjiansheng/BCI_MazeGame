"""Behavior tests for the 16-channel portable model and split boundaries."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bci_maze.lk_mini.semhaf import (
    BCIC2A_CHANNEL_NAMES, BCIC2A_GDF_NAMES, CLASSES, filter_epochs,
    fit_normalization, load_semhaf_bundle, new_bundle, public_channel_indices,
    save_semhaf_bundle,
)
from bci_maze.lk_mini.semhaf_training import (
    TrainingConfig, evaluate_public, finetune_personal, fit_model,
    load_personal_trials, load_public_trials, personal_splits, pretrain_public,
)
from bci_maze.lk_mini.simulation import save_synthetic_mi_recording
from bci_maze.lk_mini.source import DEFAULT_CHANNEL_NAMES


@pytest.fixture(autouse=True)
def small_cpu_pool():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def epochs(count=4, seed=12):
    return np.random.default_rng(seed).normal(0, 2, (count, 16, 1000)).astype(np.float32)


def test_verified_montage_order_and_channel_rejection():
    expected = [1, 5, 7, 9, 11, 13, 15, 17, 0, 2, 3, 4, 6, 12, 14, 16]
    assert public_channel_indices(BCIC2A_CHANNEL_NAMES) == expected
    assert public_channel_indices(BCIC2A_GDF_NAMES) == expected
    reversed_names = BCIC2A_CHANNEL_NAMES[::-1]
    assert public_channel_indices(reversed_names) == [21 - index for index in expected]
    with pytest.raises(ValueError, match="montage"):
        public_channel_indices(["unknown"] * 22)


def test_car_removes_common_reference_and_normalization_is_train_only():
    values = epochs()
    common_reference = np.linspace(-4, 5, 1000, dtype=np.float32)[None, None, :]
    filtered = filter_epochs(values)
    np.testing.assert_allclose(filtered, filter_epochs(values + common_reference), atol=2e-6)
    mean, std = fit_normalization(filtered[:2])
    expected_mean = filtered[:2].mean(axis=(0, 3), keepdims=True, dtype=np.float64)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-7)
    assert mean.shape == std.shape == (1, 9, 16, 1)


def test_safe_checkpoint_round_trip_and_input_contract(tmp_path):
    bundle = new_bundle(device="cpu", metadata={"provenance": {"data_origin": "synthetic"}})
    bundle.mean, bundle.std = fit_normalization(filter_epochs(epochs()))
    values = epochs(1)
    expected = bundle.predict_proba(values, 250, DEFAULT_CHANNEL_NAMES)
    path = save_semhaf_bundle(bundle, tmp_path / "model.pt")
    payload = torch.load(path, weights_only=True)
    assert payload["preprocessing"]["reference"] == "CAR"
    loaded = load_semhaf_bundle(path, device="cpu")
    np.testing.assert_allclose(loaded.predict_proba(values[0], 250, DEFAULT_CHANNEL_NAMES), expected, atol=1e-6)
    assert loaded.classes == CLASSES
    assert loaded.provenance["data_origin"] == "synthetic"
    assert np.isclose(expected.sum(), 1.0)
    for rate, names, data in (
        (500, DEFAULT_CHANNEL_NAMES, values),
        (250, DEFAULT_CHANNEL_NAMES[::-1], values),
        (250, DEFAULT_CHANNEL_NAMES, values[:, :, :999]),
        (250, DEFAULT_CHANNEL_NAMES, np.full_like(values, np.nan)),
    ):
        with pytest.raises(ValueError):
            loaded.predict_proba(data, rate, names)
    payload["preprocessing"]["causal"] = False
    torch.save(payload, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="preprocessing"):
        load_semhaf_bundle(tmp_path / "wrong.pt", device="cpu")


def test_fit_updates_spatial_weights_and_ignores_validation_for_normalization():
    bundle = new_bundle(device="cpu")
    train, validation = epochs(8), epochs(4, 13) * 20
    initial = bundle.model.fbc_branch.spatial[0].weight.detach().clone()
    report = fit_model(bundle, train, np.tile([0, 1], 4), validation, np.tile([0, 1], 2),
                       TrainingConfig(epochs=1, patience=1, batch_size=4, cpu_threads=2))
    mean, std = fit_normalization(filter_epochs(train))
    np.testing.assert_allclose(bundle.mean, mean)
    np.testing.assert_allclose(bundle.std, std)
    assert all(parameter.requires_grad for parameter in bundle.model.parameters())
    assert report["epochs_completed"] == 1
    # The selected epoch can legitimately be 0; train history still proves a step.
    assert report["history"][1]["train_loss"] is not None
    if report["best_epoch"] == 1:
        assert not torch.equal(initial, bundle.model.fbc_branch.spatial[0].weight)


def _public_file(root, subject, session="T"):
    raw = np.broadcast_to(np.arange(22, dtype=np.float32)[None, :, None], (12, 22, 1000)).copy()
    labels = np.tile([0, 1, 2, 3], 3)
    artifact = np.zeros(12, dtype=bool)
    artifact[0] = True
    np.savez(root / f"A{subject:02d}{session}.npz", raw_x=raw, y=labels, artifact=artifact,
             sfreq=np.asarray(250), channel_names=np.asarray(BCIC2A_GDF_NAMES))


def test_public_loading_selects_raw_left_right_clean_and_never_reads_E(tmp_path):
    _public_file(tmp_path, 1)
    data = load_public_trials(tmp_path, [1])
    assert data["epochs"].shape == (5, 16, 1000)
    assert data["labels"].tolist() == [1, 0, 1, 0, 1]
    assert data["epochs"][0, :, 0].tolist() == public_channel_indices(BCIC2A_GDF_NAMES)
    with pytest.raises(ValueError, match="disjoint"):
        pretrain_public(tmp_path, [1, 2], [2], TrainingConfig(epochs=1), device="cpu")
    assert not list(tmp_path.glob("*E.npz"))


def test_personal_three_way_splits_keep_final_block_and_session_isolated():
    labels = np.tile([0, 1], 30)
    groups = np.repeat(["0:0", "0:1", "1:0"], 20)
    sessions = np.repeat(["first", "first", "later"], 20)
    train, validation, test, report = personal_splits(labels, groups, sessions, holdout_session="later")
    assert set(train) == set(range(20))
    assert set(validation) == set(range(20, 40))
    assert set(test) == set(range(40, 60))
    assert report["final_holdout_used_for_training_or_selection"] is False
    # A two-block short collection splits only its first block for early stopping.
    train, validation, test, _ = personal_splits(labels[:40], groups[:40], sessions[:40])
    assert set(test) == set(range(20, 40))
    assert set(train).isdisjoint(validation) and set(train).isdisjoint(test)
    assert set(train) | set(validation) == set(range(20))
    with pytest.raises(ValueError, match="at least two blocks"):
        personal_splits(labels[:20], groups[:20], sessions[:20])
    with pytest.raises(ValueError, match="at least 5"):
        personal_splits(labels[:12], np.repeat(["0:0", "0:1"], 6), sessions[:12])


def test_synthetic_finetune_provenance_and_independent_before_after(tmp_path):
    path = save_synthetic_mi_recording(tmp_path / "synthetic.npz",
        schedule=[(block, label) for block in range(3) for _ in range(4) for label in CLASSES],
        subject_id="SYN", session_id="first")
    loaded = load_personal_trials([path], "SYN")
    assert loaded["data_origin"] == "synthetic"
    public = new_bundle(device="cpu", metadata={"provenance": {"data_origin": "public"}})
    pretrained = save_semhaf_bundle(public, tmp_path / "public.pt")
    bundle = finetune_personal(pretrained, [path], "SYN",
        TrainingConfig(epochs=1, patience=1, batch_size=4, cpu_threads=2), device="cpu")
    assert bundle.subject_id == "SYN"
    assert bundle.provenance == {"data_origin": "synthetic", "human_mi_verified": False, "calibration_passed": False}
    assert bundle.training_report["before_finetuning"]["n_trials"] == 8
    assert bundle.training_report["after_finetuning"]["n_trials"] == 8
    assert bundle.training_report["final_holdout_refit"] is False
    reloaded = load_semhaf_bundle(save_semhaf_bundle(bundle, tmp_path / "personal.pt"), "cpu")
    assert reloaded.subject_id == "SYN"
    with pytest.raises(ValueError, match="does not belong"):
        load_personal_trials([path], "ANOTHER")


def test_public_evaluation_reports_seen_and_validation_subjects_separately(tmp_path, monkeypatch):
    import bci_maze.lk_mini.semhaf_training as training
    bundle = new_bundle(device="cpu", metadata={"train_subjects": [1], "validation_subjects": [2]})
    calls = []
    def fake_load(root, subjects, session):
        calls.append((subjects, session))
        return {"epochs": epochs(4), "labels": np.array([0, 1, 0, 1])}
    monkeypatch.setattr(training, "load_public_trials", fake_load)
    monkeypatch.setattr(training, "evaluate_epochs", lambda *args: {"accuracy": .5, "balanced_accuracy": .5})
    report = evaluate_public(bundle, tmp_path, [1, 2, 3])
    assert [record["category"] for record in report["subjects"]] == [
        "cross_session_seen_subject", "heldout_subject_used_for_T_validation", "unseen_subject"]
    assert calls == [([1], "E"), ([2], "E"), ([3], "E")]
    assert not report["model_selection"]


def test_default_short_collection_cli_to_finetune_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    recording = tmp_path / "short.npz"
    collection = subprocess.run([
        sys.executable, "-X", "utf8", "scripts/collect_lk_mi.py", "--backend", "synthetic",
        "--simulate-mi", "--subject", "SHORT", "--session", "short01", "--output", str(recording),
    ], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert collection.returncode == 0, collection.stdout + collection.stderr
    with np.load(recording, allow_pickle=False) as data:
        assert data["epochs"].shape == (20, 16, 1000)
    public = new_bundle(device="cpu", metadata={"provenance": {"data_origin": "public"}})
    pretrained = save_semhaf_bundle(public, tmp_path / "public_fixture.pt")
    output = tmp_path / "short_personal.pt"
    calibration = subprocess.run([
        sys.executable, "-X", "utf8", "scripts/finetune_lk_semhaf.py", str(recording),
        "--pretrained", str(pretrained), "--subject", "SHORT", "--output", str(output),
        "--epochs", "1", "--patience", "1", "--device", "cpu", "--cpu-threads", "2",
    ], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert calibration.returncode == 0, calibration.stdout + calibration.stderr
    bundle = load_semhaf_bundle(output, "cpu")
    split = bundle.training_report["split"]
    assert split["trial_counts"] == {"train": 6, "validation": 4, "final_holdout": 10}
    assert split["class_counts"]["train"] == {"left_hand": 3, "right_hand": 3}
    assert split["small_sample"] is True
    assert bundle.training_report["before_finetuning"]["n_trials"] == 10
    assert bundle.training_report["after_finetuning"]["n_trials"] == 10
    assert bundle.provenance["data_origin"] == "synthetic"
    assert bundle.provenance["calibration_passed"] is False


def test_personal_loader_drops_damaged_artifact_and_rejects_source_pollution(tmp_path):
    path = save_synthetic_mi_recording(tmp_path / "recording.npz",
        schedule=[(block, label) for block in range(2) for _ in range(5) for label in CLASSES],
        subject_id="CHECK", session_id="one")
    with np.load(path, allow_pickle=False) as data:
        original = {key: data[key] for key in data.files}
    damaged = {key: value.copy() for key, value in original.items()}
    damaged["epochs"][0] = np.nan
    damaged["artifact"][0] = True
    np.savez(path, **damaged)
    assert len(load_personal_trials([path], "CHECK")["epochs"]) == 19
    # A fractional rate must not be silently truncated to the required 250 Hz.
    altered = {**original, "sample_rate": np.asarray(250.5)}
    np.savez(path, **altered)
    with pytest.raises(ValueError, match="exactly 250"):
        load_personal_trials([path], "CHECK")
    for update, pattern in (({"kind": "replay"}, "provenance"),
                            ({"kind": "hardware", "backend": "synthetic_mi"}, "conflicting"),
                            ({"kind": "hardware", "backend": "brainflow", "label_source": "scripted_simulation"}, "simulated labels")):
        metadata = json.loads(str(original["metadata_json"].item()))
        metadata["provenance"].update(update)
        np.savez(path, **{**original, "metadata_json": np.asarray(json.dumps(metadata))})
        with pytest.raises(ValueError, match=pattern):
            load_personal_trials([path], "CHECK")
    metadata = json.loads(str(original["metadata_json"].item()))
    metadata["units"] = "V"
    np.savez(path, **{**original, "metadata_json": np.asarray(json.dumps(metadata))})
    with pytest.raises(ValueError, match="microvolts"):
        load_personal_trials([path], "CHECK")

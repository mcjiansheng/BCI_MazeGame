import json
import sys

import numpy as np
import pytest

from bci_maze.lk_mini.game_control import model_control_provenance
from bci_maze.lk_mini.model import load_model_bundle
from bci_maze.lk_mini.simulation import save_synthetic_mi_recording
from scripts import train_lk_mi


def test_training_rejects_recording_from_another_subject(monkeypatch, tmp_path):
    recording = save_synthetic_mi_recording(
        tmp_path / "other.npz",
        schedule=[(0, "left_hand"), (0, "right_hand")] * 2,
        subject_id="S02",
        session_id="test",
        sample_rate=250,
        epoch_seconds=1.0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_lk_mi.py", str(recording), "--subject", "S01"],
    )
    with pytest.raises(ValueError, match="do not match"):
        train_lk_mi.main()


def make_training_recording(path, backend, task="motor_imagery", *, reverse_last_block=False):
    blocks = np.repeat([0, 1], 8)
    labels = np.tile(["left_hand", "right_hand"], 8)
    epochs = np.random.default_rng(42).normal(size=(16, 2, 250))
    signal = 20 * np.sin(2 * np.pi * 10 * np.arange(250) / 250)
    for trial, label in enumerate(labels):
        channel = int(label == "right_hand") ^ int(reverse_last_block and blocks[trial] == 1)
        epochs[trial, channel] += signal
    np.savez(
        path, epochs=epochs, labels=labels, blocks=blocks, artifact=np.zeros(16, dtype=bool),
        sample_rate=250, channel_names=["C3", "C4"], metadata_json=json.dumps({
            "subject_id": "S01", "configuration": {"backend": backend, "task": task}
        }),
    )
    return path


def test_training_rejects_mixed_recording_sources(monkeypatch, tmp_path):
    recorded = make_training_recording(tmp_path / "recorded.npz", "brainflow")
    synthetic = make_training_recording(tmp_path / "synthetic.npz", "synthetic_mi")
    monkeypatch.setattr(sys, "argv", ["train_lk_mi.py", str(recorded), str(synthetic), "--subject", "S01"])
    with pytest.raises(ValueError, match="single data origin"):
        train_lk_mi.main()


def test_training_rejects_a_recording_repeated_through_a_path_alias(monkeypatch, tmp_path):
    recorded = make_training_recording(tmp_path / "recorded.npz", "brainflow")
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()
    alias = subfolder / ".." / recorded.name
    output = tmp_path / "csp.joblib"
    monkeypatch.setattr(
        sys, "argv", ["train_lk_mi.py", str(recorded), str(alias), "--subject", "S01", "--output", str(output)]
    )
    with pytest.raises(ValueError, match="unique recording paths"):
        train_lk_mi.main()
    assert not output.exists()


def test_training_rejects_motor_execution_recording(monkeypatch, tmp_path):
    path = make_training_recording(tmp_path / "motor.npz", "brainflow", "motor_execution")
    monkeypatch.setattr(sys, "argv", ["train_lk_mi.py", str(path), "--subject", "S01"])
    with pytest.raises(ValueError, match="motor_execution"):
        train_lk_mi.main()


@pytest.mark.parametrize(("backend", "origin", "weak", "passed"), [
    ("brainflow", "hardware_recorded", False, True),
    ("brainflow", "hardware_recorded", True, False),
    ("lsl", "lsl_recorded", True, False),
    ("synthetic_mi", "synthetic", False, False),
])
def test_training_preserves_source_task_and_calibration(monkeypatch, tmp_path, backend, origin, weak, passed):
    path = make_training_recording(tmp_path / "recording.npz", backend, reverse_last_block=weak)
    output = tmp_path / "csp.joblib"
    monkeypatch.setattr(sys, "argv", ["train_lk_mi.py", str(path), "--subject", "S01", "--output", str(output)])
    assert train_lk_mi.main() == 0
    bundle = load_model_bundle(output)
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["above_chance_95_percent"] is not weak
    assert bundle.provenance["data_origin"] == report["data_origin"] == origin
    assert bundle.provenance["task"] == report["task"] == "motor_imagery"
    assert bundle.provenance["calibration_passed"] is report["calibration_passed"] is passed
    control = model_control_provenance(bundle, allow_unvalidated=origin == "synthetic")
    assert control["control_validated"] is passed

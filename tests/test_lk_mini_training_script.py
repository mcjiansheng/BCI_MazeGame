import sys

import pytest

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

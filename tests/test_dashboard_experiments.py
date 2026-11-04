"""Experiment task ownership, data-path isolation and report access checks."""
import json
import os
from pathlib import Path

import pytest

from bci_maze.dashboard.controller import DashboardController
from bci_maze.dashboard.experiments import list_reports, report_path


@pytest.fixture
def controller(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_lk_experiment.py").touch()
    runtime = tmp_path / ".venv-mi" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    recording = tmp_path / "data/recordings/S01/input.npz"
    recording.parent.mkdir(parents=True)
    recording.touch()
    return DashboardController(tmp_path)


def test_demo_never_requests_a_live_source(controller):
    command, output = controller._build_job(dict(kind="experiment_demo", experiment="imagery", subject="DEMO"))
    assert command[command.index("--action") + 1] == "demo"
    assert "--backend" not in command and "--ip-address" not in command
    assert output.is_relative_to(controller.root / "outputs/experiments/DEMO")
    _, second_output = controller._build_job(dict(kind="experiment_demo", experiment="imagery", subject="DEMO"))
    assert second_output != output


def test_experiment_collection_preserves_selected_recovery_profile(controller):
    command, _ = controller._build_job(dict(kind="experiment_collect", experiment="motor", subject="S01", recovery_profile="fast", wear_state="worn"))
    assert command[command.index("--no-data-timeout") + 1] == "1.5"
    assert command[command.index("--backend") + 1] == "brainflow"
    assert command[command.index("--experiment") + 1] == "motor"
    assert command[command.index("--wear-state") + 1] == "worn"


@pytest.mark.parametrize("changes", [
    {"experiment": "benchmark"}, {"backend": "replay", "replay": "data/recordings/S01/input.npz"},
    {"trials_per_class": 5, "blocks": 2}, {"subject": "../S01"}, {"permutations": 100000},
    {"wear_state": "connected"},
])
def test_reject_invalid_experiment_capture(controller, changes):
    with pytest.raises(ValueError):
        controller._build_job({"kind": "experiment_collect", "subject": "S01", "experiment": "imagery", **changes})


def test_analysis_rejects_repeated_or_external_recordings(controller):
    for recordings in (["data/recordings/S01/input.npz"] * 2, ["../private.npz"]):
        with pytest.raises(ValueError):
            controller._build_job(dict(kind="experiment_analyze", experiment="benchmark", subject="S01", recordings=recordings))


def test_report_inventory_and_asset_access_are_scoped(controller):
    root = controller.root
    folder = root / "outputs/experiments/S01/example"
    folder.mkdir(parents=True)
    report = folder / "report.json"
    report.write_text(json.dumps(dict(subject="S01", task="motor_imagery", source_kind="synthetic")), encoding="utf-8")
    (folder / "ersp.png").write_bytes(b"png-fixture")
    relative = report.relative_to(root).as_posix()
    assert list_reports(root)[0]["path"] == relative
    assert report_path(root, relative) == report
    assert report_path(root, str((folder / "ersp.png").relative_to(root)), asset=True).name == "ersp.png"
    for value, asset in (("data/recordings/S01/input.npz", False), (relative, True), ("../../private/report.json", False)):
        with pytest.raises(ValueError):
            report_path(root, value, asset=asset)

import json
import sys

import numpy as np
import pytest

from scripts import collect_lk_mi
from bci_maze.lk_mini.recording import TrialRecord, save_mi_recording


def parse_plan(arguments=()):
    args = collect_lk_mi.build_argument_parser().parse_args(["--subject", "TEST", *arguments])
    collect_lk_mi.apply_collection_preset(args)
    return args


@pytest.mark.parametrize("preset,per_class,blocks,total,expected,lower,upper", [
    ("short", 10, 2, 20, 225, 215, 235),
    ("standard", 20, 4, 40, 475, 455, 495),
    ("full", 40, 4, 80, 855, 815, 895),
])
def test_presets_preserve_four_second_windows_and_account_for_every_rest(
    preset, per_class, blocks, total, expected, lower, upper,
):
    args = parse_plan(["--preset", preset])
    schedule = collect_lk_mi.make_schedule(("left_hand", "right_hand"), args.blocks, args.trials_per_class, args.seed)
    timing = collect_lk_mi.planned_timing(args, len(schedule))
    assert (args.trials_per_class, args.blocks) == (per_class, blocks)
    assert args.imagery_seconds == 4
    assert len(schedule) == total
    assert timing["expected_seconds"] == expected
    assert timing["minimum_seconds"] == lower
    assert timing["maximum_seconds"] == upper
    assert lower <= timing["seeded_seconds"] <= upper


def test_default_short_and_explicit_cli_values_take_precedence():
    args = parse_plan()
    assert (args.preset, args.trials_per_class, args.blocks) == ("short", 10, 2)
    assert not args.preset_overridden
    args = parse_plan(["--preset", "full", "--trials-per-class", "6", "--blocks", "3"])
    assert (args.trials_per_class, args.blocks) == (6, 3)
    assert args.preset_overridden
    args = parse_plan(["--preset", "standard", "--blocks", "2"])
    assert (args.trials_per_class, args.blocks) == (20, 2)
    assert collect_lk_mi.planned_timing(args, 40)["expected_seconds"] == 415


def test_invalid_explicit_block_count_still_fails_before_device_open(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["collect", "--subject", "TEST", "--trials-per-class", "10", "--blocks", "4"])
    monkeypatch.setattr(collect_lk_mi, "source_from_args", lambda _: pytest.fail("opened device"))
    with pytest.raises(ValueError, match="divisible"):
        collect_lk_mi.main()


def test_synthetic_cli_records_plan_units_provenance_and_complete_counts(monkeypatch, tmp_path, capsys):
    path = tmp_path / "short-synthetic.npz"
    monkeypatch.setattr(sys, "argv", [
        "collect", "--subject", "TEST", "--backend", "synthetic", "--simulate-mi",
        "--output", str(path),
    ])
    assert collect_lk_mi.main() == 0
    output = capsys.readouterr().out
    assert "共 20 次 / 2 组" in output and "3 分 45 秒" in output
    with np.load(path, allow_pickle=False) as recorded:
        assert recorded["epochs"].shape == (20, 16, 1000)
        metadata = json.loads(recorded["metadata_json"].item())
    assert metadata["units"] == "uV"
    assert metadata["provenance"]["kind"] == "synthetic"
    assert metadata["provenance"]["label_source"] == "scripted_simulation"
    assert metadata["provenance"]["human_mi_verified"] is False
    assert metadata["reference"] == "synthetic"
    assert metadata["configuration"]["preset"] == "short"
    assert metadata["configuration"]["warmup_seconds"] == 5
    assert metadata["trial_summary"]["planned"] == 20
    assert metadata["trial_summary"]["usable"] == 20
    assert metadata["trial_summary"]["discarded"] == 0


@pytest.mark.parametrize("location", [None, "cap_REF"])
def test_hardware_settings_are_explicit_without_inventing_physical_reference(location):
    args = parse_plan(["--reference-location", location] if location else [])
    config = collect_lk_mi.collection_configuration(args, collect_lk_mi.planned_timing(args, 20), 20)
    assert config["reference"] == "SRB2"
    assert config["reference_location"] == (location or "unspecified")
    assert config["bias_location"] == "unspecified"
    assert config["channel_settings"]["gain"] == 24
    assert config["channel_settings"]["srb2_enabled"] is True
    assert config["channel_settings"]["srb1_enabled"] is False
    assert config["channel_settings"]["physical_montage_verified"] is False


def test_saved_hardware_metadata_separates_source_labels_and_trial_disposition(tmp_path):
    args = parse_plan(["--reference-location", "cap_REF", "--bias-location", "cap_GND"])
    config = collect_lk_mi.collection_configuration(args, {}, 20)
    path = save_mi_recording(
        tmp_path / "hardware-contract.npz", continuous=np.zeros((2, 20)),
        timestamps=np.arange(20) / 10, epochs=np.zeros((2, 2, 10)),
        labels=["left_hand", "right_hand"],
        trials=[TrialRecord(0, 0, "left_hand", 0, 10, 0), TrialRecord(1, 0, "right_hand", 10, 20, 1, artifact=True)],
        sample_rate=10, channel_names=("C3", "C4"), subject_id="TEST", session_id="test",
        configuration=config, discarded_trials=[{"trial_index": 2, "reason": "stream_interrupted"}],
    )
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata_json"].item())
    assert metadata["provenance"]["kind"] == "hardware"
    assert metadata["provenance"]["label_source"] == "visual_cue"
    assert metadata["reference_location"] == "cap_REF"
    assert metadata["bias_location"] == "cap_GND"
    assert metadata["trial_summary"] == {
        "planned": 20, "completed": 2, "usable": 1, "artifact_marked": 1,
        "discarded": 1, "unattempted": 17,
        "usable_definition": "completed trials without an operator artifact mark",
    }


def test_replay_metadata_retains_upstream_source_and_stays_replay(tmp_path):
    input_path = tmp_path / "input.npz"
    original = {"provenance": {"kind": "synthetic"}, "reference": "synthetic"}
    np.savez(input_path, metadata_json=json.dumps(original))
    args = parse_plan(["--backend", "replay", "--replay", str(input_path)])
    config = collect_lk_mi.collection_configuration(args, {}, 20)
    assert config["backend"] == "replay"
    assert config["source_provenance"] == original["provenance"]
    assert config["source_reference"] == "synthetic"
    assert config["reference"] == "unspecified"

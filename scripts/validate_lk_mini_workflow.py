"""Run the complete LK-Mini MI workflow without requiring human EEG hardware.

The validation invokes the same command-line entry points used for a real
experiment: source/channel check, labeled collection, subject model training,
and streaming replay classification. Synthetic data is explicitly marked and
is used only to validate software integration, never physiological validity.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.lk_mini.online import ProbabilitySmoother, classify_mi_window
from bci_maze.lk_mini.model import load_model_bundle
from bci_maze.lk_mini.recording import load_mi_recordings
from bci_maze.lk_mini.source import DEFAULT_CHANNEL_NAMES


def _run_stage(name: str, arguments: list[str], timeout: float = 120.0) -> None:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    current_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not current_python_path
        else source_path + os.pathsep + current_python_path
    )
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def run_workflow(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_check = output / "device_check.npz"
    calibration = output / "calibration.npz"
    validation = output / "online_validation.npz"
    model_path = output / "csp_lda.joblib"
    decisions_path = output / "online_decisions.jsonl"
    report_path = output / "workflow_report.json"
    decisions_path.unlink(missing_ok=True)

    _run_stage(
        "source and channel check",
        [
            "scripts/lk_mini_check.py",
            "--backend",
            "synthetic",
            "--no-realtime",
            "--no-plot",
            "--duration",
            "2",
            "--output",
            str(device_check),
        ],
    )
    with np.load(device_check, allow_pickle=False) as checked:
        check_data = np.asarray(checked["data"])
        check_timestamps = np.asarray(checked["timestamps"])
        check_quality = json.loads(str(np.asarray(checked["quality_json"]).item()))
        check_rate = int(np.asarray(checked["sample_rate"]).item())
        check_channels = tuple(str(value) for value in checked["channel_names"].tolist())
        check_gap_count = int(np.asarray(checked["timestamp_gap_count"]).item())

    common_collection_args = [
        "--backend",
        "synthetic",
        "--simulate-mi",
        "--classes",
        "left_hand,right_hand",
        "--imagery-seconds",
        "4",
    ]
    _run_stage(
        "synthetic MI calibration collection",
        [
            "scripts/collect_lk_mi.py",
            "--subject",
            "WORKFLOW",
            "--session",
            "calibration",
            "--trials-per-class",
            "12",
            "--blocks",
            "4",
            "--seed",
            "42",
            "--output",
            str(calibration),
            *common_collection_args,
        ],
    )
    _run_stage(
        "subject-specific CSP+LDA training",
        [
            "scripts/train_lk_mi.py",
            str(calibration),
            "--subject",
            "WORKFLOW",
            "--output",
            str(model_path),
            "--seed",
            "42",
        ],
    )

    _run_stage(
        "independent online validation collection",
        [
            "scripts/collect_lk_mi.py",
            "--subject",
            "WORKFLOW",
            "--session",
            "online-validation",
            "--trials-per-class",
            "4",
            "--blocks",
            "2",
            "--seed",
            "314",
            "--output",
            str(validation),
            *common_collection_args,
        ],
    )
    _run_stage(
        "streaming replay classification",
        [
            "scripts/online_lk_mi.py",
            "--model",
            str(model_path),
            "--backend",
            "replay",
            "--replay",
            str(validation),
            "--no-realtime",
            "--duration",
            "0",
            "--step-seconds",
            "4",
            "--history",
            "1",
            "--confidence",
            "0.5",
            "--margin",
            "0",
            "--log",
            str(decisions_path),
        ],
    )

    training_report = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    validation_data = load_mi_recordings([validation])
    expected_labels = validation_data["labels"].tolist()
    decisions = [
        json.loads(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predicted_labels = [row["label"] for row in decisions]
    online_accuracy = (
        float(np.mean(np.asarray(predicted_labels) == np.asarray(expected_labels)))
        if len(predicted_labels) == len(expected_labels) and expected_labels
        else 0.0
    )

    bundle = load_model_bundle(model_path)
    rejected = classify_mi_window(
        bundle,
        ProbabilitySmoother(bundle.classes, history=1),
        np.zeros(
            (
                len(bundle.channel_names),
                int(round(bundle.config.epoch_seconds * bundle.config.sample_rate)),
            )
        ),
        bundle.config.sample_rate,
        bundle.channel_names,
    )

    checks = {
        "sixteen_channel_default_montage": (
            len(check_channels) == 16 and check_channels == DEFAULT_CHANNEL_NAMES
        ),
        "source_shape_correct": check_data.shape == (len(check_channels), check_rate * 2),
        "source_timestamps_strictly_increasing": bool(
            check_timestamps.size > 1 and np.all(np.diff(check_timestamps) > 0)
        ),
        "source_has_no_timestamp_gaps": check_gap_count == 0,
        "all_source_channels_usable": all(item["status"] != "bad" for item in check_quality),
        "training_above_chance_95_percent": bool(
            training_report["above_chance_95_percent"]
        ),
        "model_metadata_matches_source": (
            tuple(bundle.channel_names) == check_channels
            and bundle.config.sample_rate == check_rate
        ),
        "online_processed_every_validation_trial": len(decisions) == len(expected_labels),
        "online_all_windows_accepted": bool(decisions) and all(
            row["accepted"] for row in decisions
        ),
        "online_accuracy_at_least_90_percent": online_accuracy >= 0.9,
        "flat_signal_rejected": (
            not rejected.decision.accepted
            and rejected.decision.reason == "bad_signal"
            and set(rejected.bad_channels) == set(bundle.channel_names)
        ),
    }
    passed = all(checks.values())
    report = {
        "status": "passed" if passed else "failed",
        "scope": "synthetic software integration; no claim about human MI validity",
        "device_contract": {
            "channel_count": len(check_channels),
            "sample_rate": check_rate,
            "channel_names": list(check_channels),
            "samples_checked": int(check_data.shape[1]),
            "quality": check_quality,
        },
        "training": {
            "trials": training_report["n_trials"],
            "cross_validated_accuracy": training_report["accuracy"],
            "chance_accuracy_95_percent": training_report[
                "chance_accuracy_95_percent"
            ],
            "temporal_holdout": training_report["temporal_holdout"],
        },
        "online": {
            "validation_trials": len(expected_labels),
            "decisions": len(decisions),
            "accuracy": online_accuracy,
            "accepted": sum(bool(row["accepted"]) for row in decisions),
        },
        "checks": checks,
        "artifacts": {
            "device_check": str(device_check),
            "calibration_recording": str(calibration),
            "model": str(model_path),
            "model_report": str(model_path.with_suffix(".json")),
            "online_validation_recording": str(validation),
            "online_decisions": str(decisions_path),
        },
    }
    report["artifacts"]["workflow_report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "lk_mini_workflow",
    )
    args = parser.parse_args()
    report = run_workflow(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

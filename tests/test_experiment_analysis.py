"""Scientific direction checks, group isolation and complete permutation refits."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from bci_maze.experiments.analysis import analyze_recordings, compute_analysis, load_recordings, signed_r_squared
from bci_maze.experiments import benchmark


def write_recording(path: Path, *, baseline=True, task="motor_execution", subject="S01",
                    session="session1", source="synthetic", seed=5, artifacts=(), nonfinite=None):
    rng = np.random.default_rng(seed)
    rate, n = 100, 24
    labels = np.tile(["left_hand", "right_hand"], n // 2)
    blocks = np.repeat(np.arange(3), n // 3)
    channels = np.asarray(["C3", "Cz", "C4", "FCz", "CPz"])
    active, rest = [], []
    t, b = np.arange(4 * rate) / rate, np.arange(2 * rate) / rate
    for label in labels:
        phases = rng.uniform(0, 2 * np.pi, len(channels))[:, None]
        activity = 8 * np.sin(2 * np.pi * 10 * t + phases) + 4 * np.sin(2 * np.pi * 20 * t + phases)
        resting = 8 * np.sin(2 * np.pi * 10 * b + phases) + 4 * np.sin(2 * np.pi * 20 * b + phases)
        activity[2 if label == "left_hand" else 0] *= 0.4
        active.append(activity + rng.normal(0, 0.6, activity.shape))
        rest.append(resting + rng.normal(0, 0.6, resting.shape))
    epochs, baseline_epochs = np.asarray(active), np.asarray(rest)
    flags = np.zeros(n, dtype=bool)
    flags[list(artifacts)] = True
    if nonfinite == "active":
        epochs[1, 0, 0] = np.nan
    if nonfinite == "baseline":
        baseline_epochs[1, 0, 0] = np.inf
    metadata = dict(subject_id=subject, session_id=session, units="uV",
                    provenance={"kind": source, "source_kind": source},
                    configuration={"task": task, "backend": "synthetic_mi" if source == "synthetic" else "brainflow"})
    payload = dict(epochs=epochs, labels=labels, blocks=blocks, artifact=flags, channel_names=channels,
                   sample_rate=rate, metadata_json=np.asarray(json.dumps(metadata)))
    if baseline:
        payload["baseline_epochs"] = baseline_epochs
    np.savez_compressed(path, **payload)
    return path


def test_contralateral_mu_beta_suppression_has_expected_erd_and_ersp_direction(tmp_path):
    data = load_recordings([write_recording(tmp_path / "known.npz")])
    result = compute_analysis(data)
    rows = {(row["class"], row["channel"], row["band"]): row for row in result["erd_summary"]}
    for label, contralateral, ipsilateral in (("left_hand", "C4", "C3"), ("right_hand", "C3", "C4")):
        for band in ("mu", "beta"):
            assert -90 < rows[label, contralateral, band]["erd_percent"] < -75
            assert abs(rows[label, ipsilateral, band]["erd_percent"]) < 5
        panel = next(item for item in result["ersp"]["panels"] if item["class"] == label and item["channel"] == contralateral)
        frequency = np.argmin(np.abs(result["ersp"]["frequencies"] - 10))
        assert np.mean(panel["db"][frequency]) < -6
    frequency = np.argmin(np.abs(result["frequencies"] - 10))
    assert result["signed_r2"]["values"][0, frequency] < -0.8
    assert result["signed_r2"]["values"][2, frequency] > 0.8


def test_r_squared_reverses_with_label_swap_and_bounds_constant_features():
    labels = np.asarray(["left_hand", "left_hand", "right_hand", "right_hand"])
    values = np.asarray([[1, 4, 9], [2, 3, 9], [3, 2, 9], [4, 1, 9]], dtype=float)
    first = signed_r_squared(values, labels)
    swapped = np.where(labels == "left_hand", "right_hand", "left_hand")
    np.testing.assert_allclose(signed_r_squared(values, swapped), -first)
    assert 0 < first[0] <= 1 and -1 <= first[1] < 0
    assert first[2] == 0


def test_stationary_noise_ersp_is_centered_at_zero_db(tmp_path):
    recording = load_recordings([write_recording(tmp_path / "stationary.npz")])
    rng = np.random.default_rng(1729)
    n_trials, rate = 600, recording.sample_rate
    # Both intervals have the same population power; shorter baseline duration
    # must not create a systematic ERSP offset after averaging trials.
    stationary = rng.normal(0, 2, size=(n_trials, 3, 6 * rate))
    recording = replace(recording, epochs=stationary[..., 2 * rate:],
                        baseline_epochs=stationary[..., :2 * rate],
                        labels=np.tile(["left_hand", "right_hand"], n_trials // 2),
                        channels=("C3", "Cz", "C4"))
    result = compute_analysis(recording)["ersp"]
    assert len(result["panels"]) == 6
    for panel in result["panels"]:
        assert np.isfinite(panel["db"]).all()
        assert abs(np.mean(panel["db"])) < 0.15
        # Individual frequencies have fewer independent observations than the
        # broadband mean; allow 1 dB for their finite-sample variation.
        assert np.max(np.abs(np.mean(panel["db"], axis=1))) < 1.0


def test_grouped_benchmark_holds_out_whole_runs_and_refits_per_permutation(monkeypatch):
    labels = np.tile(["left_hand", "right_hand"], 12)
    runs = np.repeat(["one", "two", "three"], 8)
    blocks = np.tile(np.repeat([0, 1], 4), 3)
    epochs = np.random.default_rng(4).normal(size=(24, 3, 100))
    epochs[:, 0, 0] = np.arange(24)
    fits = []

    class ProbePipeline:
        def fit(self, x, y):
            self.members = x[:, 0, 0].astype(int)
            fits.append((self.members.copy(), y.copy()))
            return self

        def predict(self, x):
            assert not set(x[:, 0, 0].astype(int)) & set(self.members)
            return np.repeat("left_hand", len(x))

    monkeypatch.setattr(benchmark, "build_csp_lda_pipeline", lambda config: ProbePipeline())
    result = benchmark.grouped_benchmark(epochs, labels, runs, blocks, 100, permutations=np.int64(5), seed=12)
    assert result["split_method"] == "leave_one_run_out"
    assert result["n_runs"] == result["n_groups"] == result["n_splits"] == 3
    assert len(fits) == 3 * (5 + 1)
    assert any(not np.array_equal(target, labels[members]) for members, target in fits[3:])
    for members, target in fits:
        for run in np.unique(runs[members]):
            select = runs[members] == run
            assert sorted(target[select]) == sorted(labels[members][select])
    test_indices = []
    for fold in result["folds"]:
        assert not set(fold["train_groups"]) & set(fold["test_groups"])
        test_indices.extend(fold["test_indices"])
    assert sorted(test_indices) == list(range(len(labels)))
    assert result["permutation"]["p_value"] == 1
    for boolean in (True, np.bool_(True)):
        with pytest.raises(ValueError, match="整数"):
            benchmark.grouped_benchmark(epochs, labels, runs, blocks, 100, permutations=boolean)


def test_real_csp_baseline_decodes_injected_spatial_pattern(tmp_path):
    recording = load_recordings([write_recording(tmp_path / "pattern.npz")])
    result = benchmark.grouped_benchmark(recording.epochs, recording.labels, recording.runs,
        recording.blocks, recording.sample_rate, permutations=3, seed=8)
    assert result["split_method"] == "leave_one_block_out"
    assert result["n_runs"] == 1
    assert result["balanced_accuracy"] >= 0.9
    null = result["permutation"]["null_scores"]
    assert len(null) == 3 and all(0 <= score <= 1 for score in null)
    assert result["permutation"]["p_value"] == (1 + sum(score >= result["balanced_accuracy"] for score in null)) / 4


def test_no_random_trial_fallback_when_group_structure_is_insufficient():
    labels = np.tile(["left_hand", "right_hand"], 4)
    with pytest.raises(ValueError, match="至少两个"):
        benchmark.group_splits(labels, np.repeat("single", 8), np.zeros(8))
    with pytest.raises(ValueError, match="两类"):
        benchmark.group_splits(np.repeat(["left_hand", "right_hand"], 4), np.repeat("single", 8), np.repeat([0, 1], 4))


@pytest.mark.parametrize("changed", [{"subject": "S02"}, {"task": "motor_imagery"}, {"source": "hardware"}])
def test_incompatible_identity_task_and_source_are_rejected(tmp_path, changed):
    first = write_recording(tmp_path / "first.npz")
    second = write_recording(tmp_path / "second.npz", **changed)
    with pytest.raises(ValueError, match="须一致"):
        load_recordings([first, second])


def test_same_session_and_block_names_do_not_merge_distinct_recordings(tmp_path):
    paths = [write_recording(tmp_path / f"part{i}.npz", seed=i) for i in range(2)]
    recording = load_recordings(paths)
    for split, expected in (("auto", 2), ("session", 2), ("block", 6)):
        groups, folds, _ = benchmark.group_splits(recording.labels, recording.runs, recording.blocks, split)
        assert len(np.unique(groups)) == len(folds) == expected
        assert groups[0] != groups[24]
        assert all(len({recording.trial_origins[index]["file"] for index in test}) == 1 for _, test in folds)
    assert recording.trial_origins[0]["session"] == recording.trial_origins[24]["session"]


@pytest.mark.parametrize("nonfinite", ["active", "baseline"])
def test_artifact_and_nonfinite_trials_are_removed_before_all_analysis(tmp_path, nonfinite):
    recording = load_recordings([write_recording(tmp_path / "excluded.npz", artifacts=(0,), nonfinite=nonfinite)])
    assert len(recording.epochs) == len(recording.baseline_epochs) == 22
    assert recording.excluded_counts["artifact_marked"] == 1
    assert recording.excluded_counts["nonfinite_after_artifact"] == 1
    assert [item["trial_index"] for item in recording.trial_origins][:2] == [2, 3]
    assert np.isfinite(recording.epochs).all() and np.isfinite(recording.baseline_epochs).all()
    assert all(item["n_trials"] == 11 for item in compute_analysis(recording)["psd"])


def test_missing_baseline_writes_supported_figures_without_inventing_rest(tmp_path):
    path = write_recording(tmp_path / "legacy.npz", baseline=False)
    report = analyze_recordings([path], tmp_path / "report", run_benchmark=True, permutations=0)
    assert report["baseline_available"] is False
    assert report["analysis"]["ersp"]["available"] is False
    assert report["analysis"]["erd_summary"] == []
    assert report["analysis"]["signed_r2"]["available"] is True
    ids = {item["id"] for item in report["figures"]}
    assert {"power_spectra", "signed_r2", "topomap_power", "benchmark"} <= ids
    assert "ersp" not in ids and "topomap_erd" not in ids
    assert json.loads((tmp_path / "report/report.json").read_text(encoding="utf-8"))["baseline_available"] is False
    assert not (tmp_path / "report/report.json.tmp").exists()
    for figure in report["figures"]:
        assert Path(figure["file"]).name == figure["file"]
        assert (tmp_path / "report" / figure["file"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_baseline_analysis_generates_full_figures_and_keeps_group_failure_report(tmp_path):
    path = write_recording(tmp_path / "complete.npz")
    report = analyze_recordings([path], tmp_path / "full", run_benchmark=True, permutations=1, split="session")
    assert report["source_kind"] == "synthetic"
    assert report["baseline_available"] is True
    assert report["benchmark"]["status"] == "unavailable"
    assert "采集文件" in report["benchmark"]["reason"]
    assert "至少两个" in report["benchmark"]["reason"]
    assert report["benchmark"]["n_runs"] == 1
    assert report["benchmark"]["n_run_blocks"] == 3
    assert {"ersp", "erd_comparison", "power_spectra", "topomap_power", "topomap_erd", "signed_r2"} <= {item["id"] for item in report["figures"]}
    assert report["analysis"]["ersp"]["units"] == "dB"
    assert report["methods"]["ersp"]["baseline_statistic"] == "mean_log_power_equivalent_to_geometric_mean_power"
    assert "mean_baseline_time" in report["methods"]["ersp"]["formula"]
    assert "source_urls" in report["methods"]["signed_r2"]


@pytest.mark.parametrize("excluded_by", ["artifact", "nonfinite"])
def test_all_excluded_trials_write_context_report_without_statistics_or_figures(tmp_path, monkeypatch, excluded_by):
    from bci_maze.experiments import analysis, plotting

    path = write_recording(tmp_path / "excluded_all.npz", artifacts=range(24) if excluded_by == "artifact" else ())
    if excluded_by == "nonfinite":
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key] for key in saved.files}
        payload["epochs"][::2, 0, 0] = np.nan
        payload["baseline_epochs"][1::2, 0, 0] = np.inf
        np.savez_compressed(path, **payload)

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Empty records must preserve context without computing results.")

    monkeypatch.setattr(analysis, "compute_analysis", unexpected_calculation)
    monkeypatch.setattr(analysis, "grouped_benchmark", unexpected_calculation)
    monkeypatch.setattr(plotting, "plot_report", unexpected_calculation)
    directory = tmp_path / "empty_report"
    report = analyze_recordings([path], directory, run_benchmark=True)
    assert report["status"] == "no_usable_trials"
    assert report["n_trials"] == 0 and report["class_counts"] == {"left_hand": 0, "right_hand": 0}
    assert (report["subject"], report["task"], report["source_kind"], report["sample_rate"]) == ("S01", "motor_execution", "synthetic", 100)
    assert report["channels"] == ["C3", "Cz", "C4", "FCz", "CPz"]
    assert report["baseline_available"] is True
    assert report["baseline_seconds"] == 2 and report["active_seconds"] == 4
    assert report["excluded_counts"]["total_input_trials"] == report["excluded_counts"]["total_excluded"] == 24
    counter = "artifact_marked" if excluded_by == "artifact" else "nonfinite_after_artifact"
    assert report["excluded_counts"][counter] == 24
    assert report["inputs"][0]["file"] == str(path.resolve())
    assert report["inputs"][0]["session"] == "session1"
    assert report["inputs"][0]["n_trials"] == report["inputs"][0][counter] == 24
    assert report["inputs"][0]["retained_trials"] == 0
    assert report["trial_origins"] == report["figures"] == []
    assert report["analysis"]["psd"] == report["analysis"]["erd_summary"] == []
    assert report["analysis"]["ersp"]["available"] is False
    assert report["analysis"]["signed_r2"]["available"] is False
    assert report["benchmark"]["status"] == "unavailable"
    assert report["benchmark"]["n_runs"] == report["benchmark"]["n_run_blocks"] == 0
    assert "补采" in report["message"] and report["message"] in report["notes"]
    assert json.loads((directory / "report.json").read_text(encoding="utf-8")) == report
    assert not (directory / "report.json.tmp").exists()
    assert not list(directory.glob("*.png"))

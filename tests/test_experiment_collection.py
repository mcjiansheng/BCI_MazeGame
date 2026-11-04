import json
import inspect
import signal
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import welch

from bci_maze.experiments import collection
from scripts.run_lk_experiment import validate_recordings


def metadata(saved):
    return json.loads(str(saved["metadata_json"].item()))


def bandpower(values, low, high):
    frequencies, power = welch(values, fs=250, nperseg=250, axis=-1)
    return power[..., (frequencies >= low) & (frequencies <= high)].sum(axis=-1)


def test_demo_is_reproducible_aligned_and_has_contralateral_mu_beta_reduction(tmp_path):
    paths = [tmp_path / "one.npz", tmp_path / "two.npz"]
    for path in paths:
        collection.create_demo_recording(path, experiment="imagery", subject="S01", session="A", seed=24)
    with np.load(paths[0], allow_pickle=False) as saved, np.load(paths[1], allow_pickle=False) as repeated:
        assert np.array_equal(saved["continuous"], repeated["continuous"])
        assert saved["baseline_epochs"].shape == (20, 16, 500)
        assert saved["epochs"].shape == (20, 16, 1000)
        assert len(saved["artifact"]) == len(saved["labels"]) == len(saved["blocks"]) == 20
        info = metadata(saved)
        assert info["configuration"]["task"] == "motor_imagery"
        assert info["provenance"]["source_kind"] == "synthetic"
        assert info["provenance"]["label_source"] == "scripted_simulation"
        assert not info["provenance"]["human_mi_verified"]
        for i, (trial, base) in enumerate(zip(info["trials"], info["configuration"]["baseline_intervals"])):
            assert np.array_equal(saved["epochs"][i], saved["continuous"][:, trial["start_sample"]:trial["stop_sample"]])
            assert np.array_equal(saved["baseline_epochs"][i], saved["continuous"][:, base["start_sample"]:base["stop_sample"]])
            assert base["stop_sample"] + 250 == trial["start_sample"]
        names = saved["channel_names"].tolist()
        for low, high in [(8, 12), (18, 22)]:
            ratio = bandpower(saved["epochs"], low, high) / bandpower(saved["baseline_epochs"], low, high)
            for label, contra, ipsi in [("left_hand", "C4", "C3"), ("right_hand", "C3", "C4")]:
                selected = saved["labels"] == label
                assert np.median(ratio[selected, names.index(contra)]) < .4
                assert np.median(ratio[selected, names.index(ipsi)]) > .75


def test_task_metadata_and_benchmark_groups_are_explicit(tmp_path):
    motor = collection.create_demo_recording(tmp_path / "motor.npz", experiment="motor", subject="S01", session="M")
    benchmark = collection.create_demo_recording(tmp_path / "benchmark.npz", experiment="benchmark", subject="S01", session="B")
    with np.load(motor, allow_pickle=False) as saved:
        assert metadata(saved)["configuration"]["task"] == "motor_execution"
    with np.load(benchmark, allow_pickle=False) as saved:
        assert saved["epochs"].shape[0] == 40
        assert np.unique(saved["blocks"]).size == 4
        assert metadata(saved)["configuration"]["task"] == "motor_imagery"
    with pytest.raises(ValueError, match="任务"):
        validate_recordings([motor], subject="S01", experiment="benchmark")
    with pytest.raises(ValueError, match="受试者"):
        validate_recordings([benchmark], subject="S02", experiment="imagery")


def test_legacy_task_is_accepted_without_relabeling(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, metadata_json=json.dumps({"subject_id": "S01", "configuration": {}}))
    assert validate_recordings([path], subject="S01", experiment="imagery") == [path.resolve()]
    with np.load(path, allow_pickle=False) as saved:
        assert "task" not in metadata(saved)["configuration"]


def test_cross_segment_between_baseline_and_task_is_rejected():
    class Recorder:
        recovering = False
        segment_id = 1
        sample_count = 1000

        def segment(self, *args, **kwargs):
            raise AssertionError("Cross-segment extraction must be rejected before reading data")

    with pytest.raises(ValueError, match="stream_segment_changed"):
        collection.extract_trial_pair(Recorder(), (0, 0), (300, 1), 100)


@pytest.fixture
def collection_harness(tmp_path):
    source = SimpleNamespace(sample_rate=100, channel_names=("C3", "C4"), recovery_stats={}, recovery_events=[])

    class Recorder:
        error = None
        recovering = False
        segment_id = 0
        sample_count = 0
        stop_calls = 0
        nonfinite_at = None
        last_sample_age = 0.0

        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stop_calls += 1

        def cursor(self):
            return self.sample_count, self.segment_id

        def recent(self, count):
            return np.ones((2, count))

        def wait_for_samples(self, stop, timeout, *, segment_id=None):
            return self.sample_count >= stop and segment_id == self.segment_id

        def segment(self, start, stop, *, segment_id=None):
            data = np.ones((2, stop - start))
            if self.nonfinite_at is not None and start <= self.nonfinite_at < stop:
                data[0, self.nonfinite_at - start] = np.nan
            return data

        def snapshot(self):
            return self.segment(0, self.sample_count), np.arange(self.sample_count) / 100

        def snapshot_segment_ids(self):
            return np.zeros(self.sample_count, dtype=np.int64)

    recorder = Recorder(source)

    class Display:
        aborted = False
        closed = False
        trial = 0
        interrupt_second_trial = True
        interrupt_at_trial = 2
        artifact = False
        close_calls = 0

        def set_plan(self, text):
            pass

        def begin_trial(self):
            self.trial += 1

        def consume_artifact(self):
            return self.artifact

        def show(self, title, symbol, instruction, progress, duration, provider):
            recorder.sample_count += round(duration * 100)
            if self.interrupt_second_trial and title == "运动想象" and self.trial == self.interrupt_at_trial:
                raise KeyboardInterrupt
            return True

        def close(self):
            self.close_calls += 1
            self.closed = True

    display = Display()
    args = SimpleNamespace(experiment="imagery", backend="synthetic", trials_per_class=1, blocks=1,
                           seed=42, subject="S01", session="partial", recording_output=tmp_path / "partial.npz",
                           warmup_seconds=0, fullscreen=False)
    return SimpleNamespace(args=args, source=source, recorder=recorder, display=display,
                           run=lambda: collection.collect_experiment(
                               args, source_factory=lambda _: source,
                               recorder_factory=lambda _: recorder, display_factory=lambda _: display))


def test_interrupted_collection_saves_aligned_partial_baselines(collection_harness):
    display = collection_harness.display
    display.artifact = True
    result = collection_harness.run()
    assert result.interrupted and result.completed == 1 and result.discarded == 1
    assert display.closed
    with np.load(result.path, allow_pickle=False) as saved:
        assert saved["epochs"].shape == (1, 2, 400)
        assert saved["baseline_epochs"].shape == (1, 2, 200)
        assert saved["artifact"].tolist() == [True]
        info = metadata(saved)
        assert len(info["trials"]) == len(info["configuration"]["baseline_intervals"]) == 1
        assert info["configuration"]["capture_interrupted"] is True
        assert info["discarded_trials"][0]["trial_index"] == 1
        assert info["discarded_trials"][0]["reason"] == "user_interrupted"


def test_worker_stop_is_deferred_through_snapshots_until_npz_exists(collection_harness, monkeypatch):
    from bci_maze.dashboard import job_worker

    harness = collection_harness
    harness.display.interrupt_second_trial = False
    monkeypatch.setattr(job_worker, "_stop_delivered", threading.Event())
    monkeypatch.setattr(job_worker, "_deferred_notice", threading.Event())
    visited = []

    def protect_snapshot(method, name):
        def checked():
            visited.append(name)
            # Use the actual dashboard handler with the live acquisition stack.
            job_worker.stop_signal_handler(signal.SIGINT, sys._getframe())
            assert not job_worker._stop_delivered.is_set()
            assert job_worker._deferred_notice.is_set()
            return method()
        return checked

    monkeypatch.setattr(harness.recorder, "snapshot", protect_snapshot(harness.recorder.snapshot, "continuous"))
    monkeypatch.setattr(harness.recorder, "snapshot_segment_ids",
                        protect_snapshot(harness.recorder.snapshot_segment_ids, "segments"))
    result = harness.run()
    assert visited == ["continuous", "segments"]
    assert result.completed == 2 and result.path.is_file()
    with np.load(result.path, allow_pickle=False) as saved:
        assert saved["epochs"].shape == (2, 2, 400)
        assert saved["baseline_epochs"].shape == (2, 2, 200)
    assert harness.display.close_calls == harness.recorder.stop_calls == 1

    # Once protected finalization has finished, deliver the pending stop once.
    with pytest.raises(KeyboardInterrupt):
        job_worker.stop_signal_handler(signal.SIGINT, sys._getframe())
    assert job_worker._stop_delivered.is_set()
    job_worker.stop_signal_handler(signal.SIGINT, sys._getframe())
    assert harness.display.close_calls == harness.recorder.stop_calls == 1
    assert result.path.is_file()


@pytest.mark.parametrize("interrupt_count", [1, 2])
def test_finish_retries_one_interrupt_and_preserves_repeated_stop(collection_harness, monkeypatch, interrupt_count):
    harness = collection_harness
    harness.display.interrupt_second_trial = False
    snapshot = harness.recorder.snapshot
    calls = 0

    def interrupted_snapshot():
        nonlocal calls
        calls += 1
        if calls <= interrupt_count:
            raise KeyboardInterrupt
        return snapshot()

    monkeypatch.setattr(harness.recorder, "snapshot", interrupted_snapshot)
    if interrupt_count == 2:
        with pytest.raises(KeyboardInterrupt):
            harness.run()
        assert not harness.args.recording_output.exists()
    else:
        result = harness.run()
        assert result.interrupted and result.completed == 2 and result.path.is_file()
        with np.load(result.path, allow_pickle=False) as saved:
            assert metadata(saved)["configuration"]["capture_interrupted"] is True
            assert len(saved["epochs"]) == len(saved["baseline_epochs"]) == 2
    assert calls == 2
    assert harness.display.close_calls == harness.recorder.stop_calls == 1


def test_incomplete_attempt_is_retried_in_same_planned_slot(collection_harness, monkeypatch, capsys):
    h = collection_harness
    h.display.interrupt_second_trial = False
    show = h.display.show

    def missing_samples(title, *args):
        result = show(title, *args)
        if title == "运动想象" and h.display.trial == 1:
            h.recorder.sample_count -= 300
        return result

    monkeypatch.setattr(h.display, "show", missing_samples)
    result = h.run()
    assert result.summary["target_met"] and result.summary["usable_trials"] == 2
    assert result.summary["total_attempts"] == 3 and result.summary["retry_count"] == 1
    with np.load(result.path, allow_pickle=False) as saved:
        info = metadata(saved)
        attempts = info["configuration"]["attempt_history"]
        assert [item["planned_slot"] for item in attempts] == [0, 0, 1]
        assert attempts[0]["label"] == attempts[1]["label"] and attempts[0]["block"] == attempts[1]["block"]
        assert attempts[0]["active_missing_samples"] == 300
        assert [trial["trial_index"] for trial in info["trials"]] == [1, 2]
        assert len(saved["baseline_epochs"]) == len(saved["epochs"]) == 2
    events = [json.loads(line.removeprefix("EXPERIMENT_EVENT ")) for line in capsys.readouterr().out.splitlines()
              if line.startswith("EXPERIMENT_EVENT ")]
    assert events[-1]["retained_trials"] == events[-1]["target_trials"] == 2
    assert events[-1]["total_attempts"] == 3 and events[-1]["trial_index"] == 1


@pytest.mark.parametrize("invalid", ["artifact", "nonfinite"])
def test_invalid_complete_attempt_is_preserved_and_replaced(collection_harness, monkeypatch, invalid):
    h = collection_harness
    h.display.interrupt_second_trial = False
    if invalid == "artifact":
        monkeypatch.setattr(h.display, "consume_artifact", lambda: h.display.trial == 1)
    else:
        h.recorder.nonfinite_at = 1
    result = h.run()
    assert result.completed == 3 and result.summary["usable_trials"] == 2
    assert result.summary["excluded_complete_trials"] == 1
    assert result.summary[invalid + "_trials"] == 1
    with np.load(result.path, allow_pickle=False) as saved:
        assert saved["artifact"].tolist() == ([True, False, False] if invalid == "artifact" else [False] * 3)
        assert len(saved["baseline_epochs"]) == len(saved["epochs"]) == 3
        history = metadata(saved)["configuration"]["attempt_history"]
        assert history[0]["status"] == "excluded" and history[1]["planned_slot"] == 0
        summary = metadata(saved)["trial_summary"]
        assert summary["completed"] == 3 and summary["usable"] == 2
        assert summary["attempts"] == 3 and summary["retries"] == 1
        assert summary["unattempted"] == summary["unfilled"] == 0


def test_three_attempt_limit_keeps_all_invalid_trials(collection_harness):
    h = collection_harness
    h.display.interrupt_second_trial = False
    h.display.artifact = True
    result = h.run()
    assert result.completed == 6 and result.summary["total_attempts"] == 6
    assert result.summary["usable_trials"] == 0 and not result.summary["target_met"]
    assert result.summary["retry_count"] == 4 and result.summary["status"] == "target_not_met"


def test_zero_complete_trials_preserve_raw_samples_and_empty_baseline_schema(collection_harness, tmp_path):
    from bci_maze.experiments.analysis import analyze_recordings
    h = collection_harness
    h.display.interrupt_at_trial = 1
    result = h.run()
    assert result.interrupted and result.completed == 0 and result.path.is_file()
    with np.load(result.path, allow_pickle=False) as saved:
        assert saved["continuous"].shape[1] > 0
        assert saved["epochs"].shape == (0, 2, 400)
        assert saved["baseline_epochs"].shape == (0, 2, 200)
        assert len(metadata(saved)["discarded_trials"]) == 1
    report = analyze_recordings([result.path], tmp_path / "empty-analysis")
    assert report["status"] == "no_usable_trials" and report["n_trials"] == 0


def test_complete_old_segment_remains_usable_after_recovery_started():
    class Recorder:
        recovering = True
        segment_id = 2
        sample_count = 900
        def segment(self, start, stop, *, segment_id):
            assert (start, stop, segment_id) == (0, 700, 0)
            return np.ones((2, stop - start))
    baseline, epoch, interval = collection.extract_trial_pair(Recorder(), (0, 0), (300, 0), 100)
    assert baseline.shape == (2, 200) and epoch.shape == (2, 400) and interval["segment_id"] == 0


def test_stale_stream_before_active_is_retried_without_showing_active(collection_harness, monkeypatch):
    h = collection_harness
    h.display.interrupt_second_trial = False
    show = h.display.show
    active_attempts = []
    def stale_after_cue(title, *args):
        if title == "运动想象":
            active_attempts.append(h.display.trial)
        result = show(title, *args)
        if title == "方向提示" and h.display.trial == 1:
            h.recorder.last_sample_age = 1.0
        elif title == "休息":
            h.recorder.last_sample_age = 0.0
        return result
    monkeypatch.setattr(h.display, "show", stale_after_cue)
    result = h.run()
    assert active_attempts == [2, 3]
    assert result.summary["target_met"] and result.summary["incomplete_trials"] == 1
    with np.load(result.path, allow_pickle=False) as saved:
        assert metadata(saved)["discarded_trials"][0]["reason"] == "stale_stream_before_active"


@pytest.mark.parametrize("limit", ["stable_wait_timeout", "max_duration_seconds"])
def test_waiting_limits_save_raw_without_spending_attempts(collection_harness, monkeypatch, limit):
    h = collection_harness
    h.recorder.recovering = True
    setattr(h.args, limit, 1.0)
    now = [0.0]
    monkeypatch.setattr(collection.time, "monotonic", lambda: now[0])
    show = h.display.show
    quality = []
    def elapsed_show(title, symbol, instruction, progress, duration, provider):
        quality.append(provider()[0])
        result = show(title, symbol, instruction, progress, duration, provider)
        now[0] += duration
        return result
    monkeypatch.setattr(h.display, "show", elapsed_show)
    result = h.run()
    assert quality and all(value is None for value in quality)
    assert result.summary["status"] == "time_limit" and result.summary["total_attempts"] == 0
    assert result.path.is_file() and result.summary["raw_samples"] > 0
    assert h.display.close_calls == h.recorder.stop_calls == 1


def test_failed_atomic_save_keeps_existing_output(tmp_path, monkeypatch):
    original = tmp_path / "protected.npz"
    original.write_bytes(b"prior recording")
    def fail_replace(*args):
        raise OSError("disk rename failed")
    monkeypatch.setattr(collection.os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename"):
        collection.create_demo_recording(original, experiment="imagery", subject="S01", session="atomic")
    assert original.read_bytes() == b"prior recording"
    assert not list(tmp_path.glob(".protected-*.npz"))


def test_partial_cli_completion_preserves_planned_progress(tmp_path, monkeypatch, capsys):
    from scripts import run_lk_experiment as cli
    from bci_maze.experiments import analysis
    monkeypatch.setattr(cli, "collect_experiment", lambda args:
                        collection.CollectionResult(tmp_path / "partial.npz", 2, 1, True))
    monkeypatch.setattr(analysis, "analyze_recordings", lambda *args, **kwargs: {"n_trials": 2})
    code = cli.main(["--action", "collect", "--experiment", "imagery", "--subject", "S01",
                     "--session", "A", "--output-dir", str(tmp_path / "result"),
                     "--recording-output", str(tmp_path / "partial.npz"), "--trials-per-class", "4"])
    events = [json.loads(line.removeprefix("EXPERIMENT_EVENT ")) for line in capsys.readouterr().out.splitlines()
              if line.startswith("EXPERIMENT_EVENT ")]
    assert code == 130
    assert events[-1]["phase"] == "complete"
    assert events[-1]["trial_index"] == 2 and events[-1]["total_trials"] == 8


def test_zero_trial_status_report_matches_source_and_ui_schema(tmp_path, monkeypatch):
    from scripts import run_lk_experiment as cli
    monkeypatch.setattr(cli, "collect_experiment", lambda args: collection.CollectionResult(None, 0, 1, True))
    assert cli.main(["--action", "collect", "--experiment", "motor", "--subject", "S01",
                     "--session", "A", "--output-dir", str(tmp_path / "result"),
                     "--recording-output", str(tmp_path / "empty.npz")]) == 130
    report = json.loads((tmp_path / "result/report.json").read_text(encoding="utf-8"))
    assert report["source_kind"] == "hardware"
    assert report["report_version"] == 1 and report["n_trials"] == 0
    assert report["task"] == "motor_execution" and len(report["channels"]) == 16
    assert report["figures"] == [] and report["notes"]
    assert not (tmp_path / "empty.npz").exists()


def test_cli_zero_trial_report_includes_raw_path_summary_and_wear_state(collection_harness, tmp_path, monkeypatch):
    from scripts import run_lk_experiment as cli
    h = collection_harness
    h.display.interrupt_at_trial = 1
    h.args.wear_state = "worn"
    result = h.run()
    monkeypatch.setattr(cli, "collect_experiment", lambda args: result)
    directory = tmp_path / "raw-only-report"
    assert cli.main(["--action", "collect", "--experiment", "imagery", "--subject", "S01",
                     "--session", "A", "--output-dir", str(directory),
                     "--recording-output", str(result.path), "--wear-state", "worn"]) == 130
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert (cli.ROOT / report["recording_path"]).resolve() == result.path.resolve()
    assert report["collection_summary"]["complete_trials"] == report["n_trials"] == 0
    assert report["collection_summary"]["raw_samples"] > 0
    assert report["wear_state"] == "worn" and any("连续信号" in note for note in report["notes"])
    with np.load(result.path, allow_pickle=False) as saved:
        info = metadata(saved)
        assert info["configuration"]["wear_state"] == "worn"
        assert info["provenance"]["human_mi_verified"] is False


@pytest.mark.parametrize("point", ["epochs.append(epoch)", "pending.update(artifact=artifact"])
@pytest.mark.parametrize("artifact", [False, True])
def test_worker_stop_during_trial_commit_keeps_arrays_and_counts_aligned(
        collection_harness, monkeypatch, point, artifact):
    from bci_maze.dashboard import job_worker
    h = collection_harness
    h.display.interrupt_second_trial = False
    h.display.artifact = artifact
    monkeypatch.setattr(job_worker, "_stop_delivered", threading.Event())
    monkeypatch.setattr(job_worker, "_deferred_notice", threading.Event())
    source, first_line = inspect.getsourcelines(collection.collect_experiment)
    target = next(first_line + index for index, line in enumerate(source) if line.strip().startswith(point))
    injected = []

    def trace(frame, event, arg):
        if (not injected and event == "line" and frame.f_code.co_filename == collection.__file__
                and frame.f_lineno == target):
            injected.append(frame.f_lineno)
            # This lands after trials.append, or after the usable counter update.
            # Use the same handler and live stack inspection as the real worker.
            job_worker.stop_signal_handler(signal.SIGINT, frame)
            assert not job_worker._stop_delivered.is_set()
            assert job_worker._deferred_notice.is_set()
        return trace

    show = h.display.show
    def deliver_after_commit(title, *args):
        if title == "休息" and injected:
            job_worker.stop_signal_handler(signal.SIGINT, sys._getframe())
        return show(title, *args)
    monkeypatch.setattr(h.display, "show", deliver_after_commit)
    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        result = h.run()
    finally:
        sys.settrace(previous_trace)
    assert injected and job_worker._stop_delivered.is_set()
    assert result.interrupted and result.completed == 1 and result.discarded == 0
    assert result.summary["usable_trials"] == int(not artifact)
    assert result.summary["unfilled_trials"] == 2 - int(not artifact)
    with np.load(result.path, allow_pickle=False) as saved:
        assert saved["epochs"].shape == (1, 2, 400)
        assert saved["baseline_epochs"].shape == (1, 2, 200)
        assert saved["artifact"].tolist() == [artifact]
        info = metadata(saved)
        assert len(info["configuration"]["baseline_intervals"]) == len(info["trials"]) == 1
        assert info["discarded_trials"] == []
        attempts = info["configuration"]["attempt_history"]
        assert len(attempts) == 1 and attempts[0]["status"] == ("excluded" if artifact else "retained")
        assert info["trial_summary"]["usable"] == int(not artifact)
        assert info["trial_summary"]["unfilled"] == 2 - int(not artifact)
        assert info["trial_summary"]["attempts"] == 1
    job_worker.stop_signal_handler(signal.SIGINT, sys._getframe())
    assert h.display.close_calls == h.recorder.stop_calls == 1

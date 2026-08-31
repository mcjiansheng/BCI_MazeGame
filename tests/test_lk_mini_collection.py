import sys

import numpy as np
import pytest

from scripts import collect_lk_mi


def test_collection_rejects_duplicate_classes_before_opening_device(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_lk_mi.py",
            "--subject",
            "S01",
            "--classes",
            "left_hand,left_hand",
        ],
    )
    with pytest.raises(ValueError, match="distinct classes"):
        collect_lk_mi.main()


def test_collection_saves_completed_trials_before_reraising_stream_error(
    monkeypatch, tmp_path
):
    class FakeSource:
        sample_rate = 10
        channel_names = ("C3", "C4")

    class FakeDisplay:
        def __init__(self, _fullscreen):
            self.closed = False

        def begin_trial(self):
            pass

        def consume_artifact(self):
            return False

        def show(self, *_args, **_kwargs):
            return True

        def close(self):
            self.closed = True

    class FailingRecorder:
        error = None

        def __init__(self, source):
            self.source = source
            self.completed = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            pass

        @property
        def sample_count(self):
            return self.completed * self.source.sample_rate

        def recent(self, samples):
            return np.zeros((2, samples))

        def wait_for_samples(self, _stop_sample, timeout):
            assert timeout == 2.0
            if self.completed:
                raise RuntimeError("device disconnected")
            self.completed += 1
            return True

        def segment(self, start, stop):
            return np.zeros((2, stop - start))

        def snapshot(self):
            samples = self.source.sample_rate * 2
            return np.zeros((2, samples)), np.arange(samples) / self.source.sample_rate

    output = tmp_path / "partial.npz"
    fake_source = FakeSource()
    monkeypatch.setattr(collect_lk_mi, "CueDisplay", FakeDisplay)
    monkeypatch.setattr(collect_lk_mi, "source_from_args", lambda _args: fake_source)
    monkeypatch.setattr(collect_lk_mi, "ContinuousRecorder", FailingRecorder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_lk_mi.py",
            "--subject",
            "S01",
            "--trials-per-class",
            "2",
            "--blocks",
            "1",
            "--prepare-seconds",
            "0",
            "--cue-seconds",
            "0",
            "--imagery-seconds",
            "1",
            "--rest-min",
            "0",
            "--rest-max",
            "0",
            "--warmup-seconds",
            "0",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="device disconnected"):
        collect_lk_mi.main()

    with np.load(output, allow_pickle=False) as saved:
        assert saved["epochs"].shape == (1, 2, 10)
        assert int(saved["sample_rate"].item()) == 10
        assert saved["channel_names"].tolist() == ["C3", "C4"]

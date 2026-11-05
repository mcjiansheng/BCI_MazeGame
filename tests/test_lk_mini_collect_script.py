import json
import sys
from types import SimpleNamespace

import numpy as np

from scripts import collect_lk_mi


def test_collection_saves_completed_trial_and_records_interrupted_trial(monkeypatch, tmp_path):
    source = SimpleNamespace(sample_rate=10, channel_names=("C3", "C4"), recovery_stats={"reconnects": 1}, recovery_events=[])

    class Recorder:
        error = None
        recovering = False
        segment_id = 0
        sample_count = 0

        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def cursor(self):
            return self.sample_count, self.segment_id

        def recent(self, count):
            return np.ones((2, count))

        def wait_for_samples(self, stop, timeout, *, segment_id=None):
            return segment_id == self.segment_id

        def segment(self, start, stop, *, segment_id=None):
            assert segment_id == self.segment_id == 0
            return np.ones((2, stop - start))

        def snapshot(self):
            return np.ones((2, 20)), np.r_[np.arange(10) / 10, 3 + np.arange(10) / 10]

        def snapshot_received_timestamps(self):
            return np.full(self.snapshot()[1].shape, np.nan)

        def snapshot_segment_ids(self):
            return np.repeat([0, 1], 10)

    recorder = Recorder(source)

    class Display:
        aborted = False

        def __init__(self, fullscreen):
            pass

        def begin_trial(self):
            pass

        def set_plan(self, text):
            self.plan = text

        def consume_artifact(self):
            return False

        def close(self):
            pass

        def show(self, phase, *args):
            if phase == "开始运动想象":
                if recorder.sample_count:
                    recorder.segment_id = 1
                recorder.sample_count += 10
            return True

    output = tmp_path / "partial.npz"
    monkeypatch.setattr(collect_lk_mi, "source_from_args", lambda args: source)
    monkeypatch.setattr(collect_lk_mi, "ContinuousRecorder", lambda source: recorder)
    monkeypatch.setattr(collect_lk_mi, "CueDisplay", Display)
    monkeypatch.setattr(sys, "argv", ["collect", "--subject", "test", "--backend", "synthetic", "--blocks", "1", "--trials-per-class", "1", "--imagery-seconds", "1", "--output", str(output)])
    assert collect_lk_mi.main() == 3
    with np.load(output, allow_pickle=False) as saved:
        assert saved["epochs"].shape == (1, 2, 10)
        assert len(saved["labels"]) == len(saved["artifact"]) == len(saved["blocks"]) == 1
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["trials"][0]["trial_index"] == 0
        assert metadata["discarded_trials"][0]["trial_index"] == 1
        assert metadata["discarded_trials"][0]["reason"] == "stream_interrupted"

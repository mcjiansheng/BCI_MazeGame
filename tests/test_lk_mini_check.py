import sys

import numpy as np
import pytest

from bci_maze.lk_mini.source import EEGChunk, EEGSource, SyntheticSource
from scripts import lk_mini_check


@pytest.mark.parametrize("fail_at_start", [False, True])
def test_check_retains_only_valid_prefix_and_never_passes_after_failure(
    monkeypatch, tmp_path, fail_at_start
):
    class FailingSource(SyntheticSource):
        def start(self):
            if fail_at_start:
                raise RuntimeError("connection timed out")
            super().start()

        def read(self, timeout=0.25):
            if self._sample_index:
                raise RuntimeError("package discontinuity")
            return super().read(timeout)

    source = FailingSource(realtime=False, chunk_samples=500)
    output = tmp_path / "interrupted_check.npz"
    monkeypatch.setattr(lk_mini_check, "source_from_args", lambda args: source)
    monkeypatch.setattr(sys, "argv", [
        "lk_mini_check.py", "--backend", "synthetic", "--no-plot",
        "--duration", "10", "--output", str(output),
    ])
    assert lk_mini_check.main() == (2 if fail_at_start else 3)
    assert not source._running
    if fail_at_start:
        assert not output.exists()
    else:
        with np.load(output, allow_pickle=False) as saved:
            assert saved["data"].shape == (16, 500)
            assert bool(saved["capture_interrupted"])
            assert "package discontinuity" in str(saved["capture_error"])


def test_check_keeps_deadline_and_saves_recovered_segment_statistics(monkeypatch, tmp_path):
    import json

    clock = [0.0]

    class RecoveringSource(EEGSource):
        sample_rate = 250
        channel_names = ("C3", "C4")
        index = 0
        running = False

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

        @property
        def recovering(self):
            return self.index == 2

        @property
        def recovery_stats(self):
            return {"gaps": 1, "reconnects": int(self.index >= 3)}

        def read(self, timeout=0.25):
            clock[0] += timeout
            self.index += 1
            if self.index not in (1, 3):
                return EEGChunk(np.empty((2, 0)), np.empty(0), segment_id=1)
            timestamps = np.arange(500) / self.sample_rate + (3 if self.index == 3 else 0)
            signal = np.tile(np.sin(np.arange(500) * 0.3) * 20, (2, 1))
            return EEGChunk(signal, timestamps, segment_id=int(self.index == 3))

    source = RecoveringSource()
    output = tmp_path / "recovered.npz"
    monkeypatch.setattr(lk_mini_check.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(lk_mini_check, "source_from_args", lambda args: source)
    monkeypatch.setattr(sys, "argv", ["check", "--backend", "synthetic", "--no-plot", "--duration", "1", "--output", str(output)])
    assert lk_mini_check.main() == 3
    assert source.index == 4
    assert clock[0] == 1.0
    assert not source.running
    with np.load(output, allow_pickle=False) as saved:
        assert saved["segment_ids"].tolist() == [0] * 500 + [1] * 500
        assert json.loads(str(saved["recovery_stats_json"]))["reconnects"] == 1
        assert len(json.loads(str(saved["segments_json"]))) == 2
        assert int(saved["timestamp_gap_count"]) == 1

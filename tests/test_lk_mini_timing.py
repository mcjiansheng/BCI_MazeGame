import json

import numpy as np
import pytest

from bci_maze.lk_mini.model import EpochBandpass
from bci_maze.lk_mini.source import ReplaySource
from bci_maze.lk_mini.timing import StreamTimingGuard


@pytest.mark.parametrize("bad", [[0.008, 0.008], [0.008, 0.004], [0.008, 0.040], [np.nan]])
def test_stream_rejects_bad_timestamps(bad):
    guard = StreamTimingGuard(250)
    guard.check(np.asarray([0, 0.004]))
    with pytest.raises(ValueError):
        guard.check(np.asarray(bad))


def test_stream_rejects_gap_between_chunks():
    guard = StreamTimingGuard(250)
    guard.check(np.asarray([1.0, 1.004]))
    with pytest.raises(ValueError, match="discontinuity"):
        guard.check(np.asarray([1.5, 1.504]))


@pytest.mark.parametrize(("stamp", "message"), [
    (np.nan, "finite"),
    (1.003, "discontinuity"),
    (1.5, "discontinuity"),
])
def test_replay_rejects_bad_recorded_timing_across_reads(tmp_path, stamp, message):
    path = tmp_path / "gap.npz"
    np.savez(path, data=np.ones((2, 3)), sample_rate=250,
             channel_names=["C3", "C4"], timestamps=[1, 1.004, stamp])
    with ReplaySource(path, realtime=False, chunk_samples=2) as source:
        assert source.read().data.shape[1] == 2
        with pytest.raises(ValueError, match=message):
            source.read()


def test_replay_preserves_lsl_jitter_and_rejects_reverse_order(tmp_path):
    path = tmp_path / "lsl.npz"
    np.savez(path, data=np.ones((2, 3)), sample_rate=250, channel_names=["C3", "C4"],
             timestamps=[1, 1.016, 1.014],
             metadata_json=json.dumps({"configuration": {"backend": "lsl"}}))
    with ReplaySource(path, realtime=False, chunk_samples=2) as source:
        assert np.diff(source.read().monotonic_timestamps) == pytest.approx([0.016])
        with pytest.raises(ValueError, match="discontinuity"):
            source.read()


@pytest.mark.parametrize("recorded_boundary", [True, False])
def test_replay_requires_segment_for_large_lsl_gap(tmp_path, recorded_boundary):
    path = tmp_path / "lsl-gap.npz"
    np.savez(path, data=np.ones((2, 3)), sample_rate=250, channel_names=["C3", "C4"],
             timestamps=[1, 1.016, 1.516], segment_ids=[0, 0, int(recorded_boundary)],
             metadata_json=json.dumps({"configuration": {"backend": "lsl"}}))
    with ReplaySource(path, realtime=False, chunk_samples=2) as source:
        first = source.read()
        if recorded_boundary:
            second = source.read()
            assert second.segment_id == 1
            assert second.monotonic_timestamps[0] - first.monotonic_timestamps[-1] == pytest.approx(0.5)
        else:
            with pytest.raises(ValueError, match="discontinuity"):
                source.read()


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_training_rejects_corrupt_epochs(value):
    epochs = np.ones((4, 16, 1000))
    epochs[1, 15, 100] = value
    with pytest.raises(ValueError, match="NaN or infinity"):
        EpochBandpass().fit_transform(epochs)

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


def test_replay_does_not_hide_recorded_gaps(tmp_path):
    path = tmp_path / "gap.npz"
    np.savez(path, data=np.ones((2, 4)), sample_rate=250,
             channel_names=["C3", "C4"], timestamps=[1, 1.004, 1.5, 1.504])
    with ReplaySource(path, realtime=False) as source:
        chunk = source.read()
    assert np.isclose(np.diff(chunk.monotonic_timestamps)[1], 0.496)
    with pytest.raises(ValueError, match="discontinuity"):
        StreamTimingGuard(250).check(chunk.monotonic_timestamps)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_training_rejects_corrupt_epochs(value):
    epochs = np.ones((4, 16, 1000))
    epochs[1, 15, 100] = value
    with pytest.raises(ValueError, match="NaN or infinity"):
        EpochBandpass().fit_transform(epochs)

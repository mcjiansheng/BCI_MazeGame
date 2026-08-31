import time

import numpy as np

from bci_maze.lk_mini.recording import (
    ContinuousRecorder,
    TrialRecord,
    load_mi_recordings,
    save_mi_recording,
)
from bci_maze.lk_mini.source import EEGChunk, EEGSource


class FiniteSource(EEGSource):
    sample_rate = 10
    channel_names = ("C3", "C4")

    def __init__(self):
        self.position = 0
        self.running = False

    def start(self):
        self.position = 0
        self.running = True

    def read(self, timeout=0.25):
        if not self.running:
            raise RuntimeError("not started")
        if self.position >= 12:
            time.sleep(0.001)
            return EEGChunk(np.empty((2, 0)), np.empty(0))
        stop = min(self.position + 4, 12)
        samples = np.arange(self.position, stop, dtype=float)
        self.position = stop
        return EEGChunk(np.vstack((samples, samples + 100)), samples / self.sample_rate)

    def stop(self):
        self.running = False


def test_recorder_segments_across_chunks():
    with ContinuousRecorder(FiniteSource()) as recorder:
        assert recorder.wait_for_samples(12, timeout=1.0)
        assert np.array_equal(recorder.recent(5)[0], np.arange(7, 12))
        assert np.array_equal(recorder.segment(3, 10)[0], np.arange(3, 10))
        continuous, timestamps = recorder.snapshot()
    assert continuous.shape == (2, 12)
    assert timestamps.shape == (12,)


def test_recording_round_trip_and_block_offsets(tmp_path):
    paths = []
    for session in range(2):
        path = tmp_path / f"session-{session}.npz"
        trials = [
            TrialRecord(0, 0, "left_hand", 0, 10, 1.0),
            TrialRecord(1, 1, "right_hand", 10, 20, 2.0),
        ]
        save_mi_recording(
            path,
            continuous=np.zeros((2, 20)),
            timestamps=np.arange(20) / 10,
            epochs=np.zeros((2, 2, 10)),
            labels=("left_hand", "right_hand"),
            trials=trials,
            sample_rate=10,
            channel_names=("C3", "C4"),
            subject_id="S01",
            session_id=str(session),
            configuration={"backend": "synthetic"},
        )
        paths.append(path)

    loaded = load_mi_recordings(paths)
    assert loaded["epochs"].shape == (4, 2, 10)
    assert loaded["blocks"].tolist() == [0, 1, 2, 3]
    assert len(loaded["metadata"]) == 2

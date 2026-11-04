import time

import numpy as np
import pytest

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


def test_recorder_stops_at_gap_and_keeps_previous_samples():
    class GappedSource(FiniteSource):
        def read(self, timeout=0.25):
            chunk = super().read(timeout)
            if self.position > 4:
                return EEGChunk(chunk.data, chunk.monotonic_timestamps + 1.0)
            return chunk

    with ContinuousRecorder(GappedSource()) as recorder:
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            recorder.wait_for_samples(12, timeout=1)
        data, timestamps = recorder.snapshot()
    assert data.shape == (2, 4)
    assert np.allclose(timestamps, np.arange(4) / 10)


def test_recorder_keeps_recovered_segments_separate_and_rejects_cross_gap_trial():
    class RecoveringSource(FiniteSource):
        def read(self, timeout=0.25):
            chunk = super().read(timeout)
            segment = int(self.position > 4)
            return EEGChunk(chunk.data, chunk.monotonic_timestamps + segment, segment_id=segment)

    with ContinuousRecorder(RecoveringSource()) as recorder:
        assert recorder.wait_for_samples(12, timeout=1)
        assert recorder.snapshot_segment_ids().tolist() == [0] * 4 + [1] * 8
        assert recorder.recent(12).shape[1] == 8
        # The completed old segment survives a later reconnect.
        assert recorder.wait_for_samples(4, timeout=0, segment_id=0)
        assert recorder.last_sample_age is not None and recorder.last_sample_age < 1
        assert not recorder.wait_for_samples(8, timeout=0.1, segment_id=0)
        with pytest.raises(ValueError, match="segment boundary"):
            recorder.segment(2, 8)
        with pytest.raises(ValueError, match="segment boundary"):
            recorder.segment(4, 8, segment_id=0)
        assert recorder.segment(4, 8, segment_id=1).shape == (2, 4)
        data, timestamps = recorder.snapshot()
    assert data.shape == (2, 12)
    assert timestamps[4] - timestamps[3] > 1


def test_completed_endpoint_survives_later_recovery_but_missing_tail_does_not():
    class RecoveringSource(FiniteSource):
        recovering = True

    recorder = ContinuousRecorder(RecoveringSource())
    assert recorder.last_sample_age is None
    recorder._chunks = [np.ones((2, 8))]
    recorder._segment_ids = [0]
    recorder._samples = 8
    recorder.error = RuntimeError("later source failure")
    assert recorder.wait_for_samples(8, timeout=0, segment_id=0)
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        recorder.wait_for_samples(9, timeout=0, segment_id=0)
    recorder.error = None
    assert not recorder.wait_for_samples(9, timeout=0, segment_id=0)


def test_recording_persists_discarded_trials_and_recovery_metadata(tmp_path):
    import json

    path = save_mi_recording(
        tmp_path / "segmented.npz", continuous=np.ones((2, 20)),
        timestamps=np.arange(20) / 10, epochs=np.ones((1, 2, 10)),
        labels=["left_hand"], trials=[TrialRecord(0, 0, "left_hand", 0, 10, 0)],
        sample_rate=10, channel_names=("C3", "C4"), subject_id="test", session_id="recovery",
        configuration={}, segment_ids=np.repeat([0, 1], 10),
        discarded_trials=[{"trial_index": 1, "reason": "stream_interrupted"}],
        recovery_stats={"reconnects": 1}, recovery_events=[{"type": "reconnected"}],
    )
    with np.load(path, allow_pickle=False) as saved:
        assert saved["segment_ids"].tolist() == [0] * 10 + [1] * 10
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["discarded_trials"][0]["reason"] == "stream_interrupted"
        assert metadata["recovery_stats"]["reconnects"] == 1
    assert load_mi_recordings([path])["epochs"].shape == (1, 2, 10)

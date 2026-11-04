import json
import sys
from types import SimpleNamespace

import numpy as np

from bci_maze.lk_mini.source import EEGChunk, EEGSource
from bci_maze.lk_mini.game_control import ControlSnapshot
from scripts import online_lk_mi


def test_online_rewarms_after_empty_recovery_chunk_without_mixing_segments(monkeypatch, tmp_path):
    class RecoveringSource(EEGSource):
        sample_rate = 10
        channel_names = ("C3", "C4")

        def __init__(self):
            self.index = 0
            self.items = [(0, 4), (0, 2), (1, 0), (1, 2), (1, 2), (1, 2), (1, 2)]
            self.position = 0

        def start(self):
            pass

        def stop(self):
            pass

        @property
        def exhausted(self):
            return self.index > len(self.items)

        @property
        def recovering(self):
            return self.index == 3

        def read(self, timeout=0.25):
            self.index += 1
            if self.index > len(self.items):
                return EEGChunk(np.empty((2, 0)), np.empty(0), segment_id=1)
            segment, count = self.items[self.index - 1]
            timestamps = np.arange(self.position, self.position + count) / 10 + segment * 2
            self.position += count
            return EEGChunk(np.full((2, count), segment + 1.0), timestamps, segment_id=segment)

    source = RecoveringSource()
    bundle = SimpleNamespace(
        config=SimpleNamespace(epoch_seconds=0.4, sample_rate=10),
        channel_names=source.channel_names, classes=("left_hand", "right_hand"), subject_id="test",
    )
    seen_windows = []

    def classify(bundle, smoother, window, rate, channels):
        seen_windows.append(window.copy())
        return SimpleNamespace(decision=smoother.update([0.95, 0.05]), bad_channels=())

    log = tmp_path / "online.jsonl"
    transport_log = tmp_path / "transport.jsonl"
    sent = []
    cleanup_order = []
    class Game:
        snapshot = ControlSnapshot(phase="continuous", visit_token=(0,))
        def __init__(self, *args, **kwargs):
            self.ready = False
        def start(self):
            pass
        def set_stream_ready(self, ready):
            self.ready = ready
        def submit(self, command, token):
            assert self.ready
            sent.append(command)
            return True
        def drain_events(self):
            return []
        def close(self):
            cleanup_order.append("game")
    def stop_source():
        assert cleanup_order[-1] == "game"
        cleanup_order.append("source")
    monkeypatch.setattr(online_lk_mi, "load_model_bundle", lambda path: bundle)
    monkeypatch.setattr(online_lk_mi, "source_from_args", lambda args: source)
    monkeypatch.setattr(online_lk_mi, "classify_mi_window", classify)
    monkeypatch.setattr(online_lk_mi, "GameControlWorker", Game)
    monkeypatch.setattr(source, "stop", stop_source)
    monkeypatch.setattr(sys, "argv", ["online", "--model", "unused", "--step-seconds", "0.2", "--history", "3", "--log", str(log), "--transport-log", str(transport_log), "--unity-host", "localhost", "--allow-unvalidated-control"])
    assert online_lk_mi.main() == 0
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [record["reason"] for record in records] == [
        "warming_up", "warming_up", "stream_recovering", "warming_up", "warming_up", "accepted",
    ]
    assert [record["segment_id"] for record in records] == [0, 0, 1, 1, 1, 1]
    assert len(sent) == 1
    assert all(np.all(window == (1 if i < 2 else 2)) for i, window in enumerate(seen_windows))
    windows = [record for record in records if "window_samples" in record]
    assert all(record["window_samples"] == 4 for record in windows)
    assert windows[2]["window_start"] == 2.6
    assert windows[2]["window_end"] == 2.9
    assert windows[2]["segment_samples_received"] == 4
    transport = [json.loads(line) for line in transport_log.read_text(encoding="utf-8").splitlines()]
    assert transport[0]["event"] == "start"
    assert transport[-1]["event"] == "summary"
    assert transport[-1]["samples"] == 14
    assert transport[-1]["nonfinite_data_values"] == 0
    assert any(record.get("first_after_recovery") for record in transport)


def test_pt_checkpoint_loads_semhaf_with_explicit_device(monkeypatch, tmp_path):
    import types
    module = types.ModuleType("bci_maze.lk_mini.semhaf")
    observed = []
    module.load_semhaf_bundle = lambda path, device: observed.append((path, device)) or "semhaf"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / "subject.pt"
    assert online_lk_mi.load_classifier(path, "cuda") == "semhaf"
    assert observed == [(path, "cuda")]


def test_new_junction_discards_pre_pause_samples_before_building_classification_window(monkeypatch, tmp_path):
    class Source(EEGSource):
        sample_rate = 10
        channel_names = ("C3", "C4")
        index = 0
        position = 0
        def start(self):
            pass
        def stop(self):
            pass
        @property
        def exhausted(self):
            return self.index > 3
        def read(self, timeout=0.25):
            self.index += 1
            count = 4 if self.index <= 2 else 2 if self.index == 3 else 0
            timestamps = np.arange(self.position, self.position + count) / 10
            self.position += count
            return EEGChunk(np.tile(timestamps, (2, 1)), timestamps)

    source = Source()
    submitted = []
    class Game:
        def __init__(self, *args, **kwargs):
            pass
        @property
        def snapshot(self):
            if source.index < 2:
                return ControlSnapshot(phase="travelling", window_id=0)
            return ControlSnapshot(phase="waiting", window_id=1, visit_token=(1, 2, 3, 0, 4),
                                   navigation={"can_left": True, "can_right": True}, window_started=0.6)
        def start(self):
            pass
        def set_stream_ready(self, ready):
            pass
        def submit(self, command, token):
            submitted.append((command, token))
            return True
        def drain_events(self):
            return []
        def close(self):
            pass

    windows = []
    bundle = SimpleNamespace(config=SimpleNamespace(epoch_seconds=0.4, sample_rate=10),
                             channel_names=source.channel_names, classes=("left_hand", "right_hand"), subject_id="test")
    def classify(bundle, smoother, window, rate, channels):
        windows.append(window.copy())
        return SimpleNamespace(decision=smoother.update([0.9, 0.1]), bad_channels=())
    monkeypatch.setattr(online_lk_mi, "load_model_bundle", lambda path: bundle)
    monkeypatch.setattr(online_lk_mi, "source_from_args", lambda args: source)
    monkeypatch.setattr(online_lk_mi, "GameControlWorker", Game)
    monkeypatch.setattr(online_lk_mi, "classify_mi_window", classify)
    monkeypatch.setattr(sys, "argv", ["online", "--model", "unused", "--step-seconds", "0.2", "--history", "1", "--unity-host", "localhost", "--allow-unvalidated-control", "--log", str(tmp_path / "junction.jsonl")])
    assert online_lk_mi.main() == 0
    assert len(windows) == 2
    assert np.allclose(windows[1][0], [0.6, 0.7, 0.8, 0.9])
    assert submitted == [("left", (1, 2, 3, 0, 4))]

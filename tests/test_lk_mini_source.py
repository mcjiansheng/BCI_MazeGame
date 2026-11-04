import numpy as np
import pytest
import sys
import types

from bci_maze.lk_mini.source import (
    BrainFlowSource,
    EEGChunk,
    DEFAULT_CHANNEL_NAMES,
    LEGACY_CHANNEL_NAMES,
    LSLSource,
    ReplaySource,
    SyntheticSource,
    build_all_channel_commands,
    build_channel_command,
    build_impedance_command,
)


def test_lk_channel_commands_match_vendor_protocol():
    assert build_channel_command(0, gain=24) == "x1060110X"
    assert build_channel_command(8, gain=6) == "xQ030110X"
    assert build_impedance_command(3, p_input=True, n_input=False) == "z410Z"
    command = build_all_channel_commands((0, 1), gain=24)
    assert command.startswith("x1060110Xx2060110Xx3160110X")
    assert len(command) == 16 * 9


def test_channel_command_rejects_invalid_values():
    with pytest.raises(ValueError):
        build_channel_command(16)
    with pytest.raises(ValueError):
        build_channel_command(0, gain=16)


@pytest.mark.parametrize(
    ("source_type", "kwargs"),
    [
        (BrainFlowSource, {"channel_indices": (), "channel_names": ()}),
        (
            BrainFlowSource,
            {"channel_indices": (0, 0), "channel_names": ("C3", "C4")},
        ),
        (
            BrainFlowSource,
            {"channel_indices": (0, 16), "channel_names": ("C3", "C4")},
        ),
        (LSLSource, {"channel_indices": (-1,), "channel_names": ("C3",)}),
        (
            LSLSource,
            {"channel_indices": (0, 1), "channel_names": ("C3", "C3")},
        ),
    ],
)
def test_live_sources_reject_invalid_channel_selections(source_type, kwargs):
    with pytest.raises(ValueError):
        source_type(**kwargs)


def test_synthetic_source_returns_microvolt_like_chunks():
    source = SyntheticSource(realtime=False, seed=7)
    with source:
        first = source.read()
        second = source.read()
        chunk = first
    assert chunk.data.shape == (16, 25)
    assert chunk.monotonic_timestamps.shape == (25,)
    assert np.isfinite(chunk.data).all()
    assert np.diff(chunk.monotonic_timestamps).min() > 0
    assert second.monotonic_timestamps[0] > first.monotonic_timestamps[-1]


def test_replay_source_reports_eof_and_preserves_sample_timing(tmp_path):
    path = tmp_path / "replay.npz"
    np.savez(
        path,
        continuous=np.arange(24, dtype=float).reshape(2, 12),
        sample_rate=np.asarray(10),
        channel_names=np.asarray(["C3", "C4"]),
    )
    source = ReplaySource(path, realtime=False, chunk_samples=5)
    timestamps = []
    with source:
        while not source.exhausted:
            timestamps.extend(source.read().monotonic_timestamps.tolist())
        assert source.read().data.shape == (2, 0)
    assert len(timestamps) == 12
    assert np.allclose(np.diff(timestamps), 0.1)


def test_replay_splits_chunks_at_recovery_boundary_and_preserves_gap(tmp_path):
    path = tmp_path / "segmented.npz"
    np.savez(path, continuous=np.ones((2, 6)), sample_rate=250,
             channel_names=np.asarray(["C3", "C4"]),
             timestamps=np.asarray([10.0, 10.004, 10.008, 15.0, 15.004, 15.008]),
             segment_ids=np.asarray([0, 0, 0, 2, 2, 2]))
    with ReplaySource(path, realtime=False, chunk_samples=5) as source:
        first, second = source.read(), source.read()
        assert first.data.shape == second.data.shape == (2, 3)
        assert first.segment_id == 0 and second.segment_id == 2
        assert np.isclose(second.monotonic_timestamps[0] - first.monotonic_timestamps[-1], 4.992)
        assert source.exhausted


@pytest.mark.parametrize("segments", [[0, 1], [0, -1, 0], [0.0, 1.0, 1.0], [1, 0, 1]])
def test_replay_rejects_invalid_segments(tmp_path, segments):
    path = tmp_path / "invalid-segments.npz"
    np.savez(path, continuous=np.ones((2, 3)), sample_rate=250,
             channel_names=np.asarray(["C3", "C4"]), segment_ids=np.asarray(segments))
    with pytest.raises(ValueError, match="segment_ids"):
        ReplaySource(path).start()


def test_chunk_segment_defaults_and_validation():
    assert EEGChunk(np.ones((1, 2)), np.asarray([0, 1])).segment_id == 0
    with pytest.raises(ValueError, match="segment_id"):
        EEGChunk(np.ones((1, 2)), np.asarray([0, 1]), segment_id=-1)


@pytest.mark.parametrize("indices,names", [
    (tuple(range(16)), DEFAULT_CHANNEL_NAMES),
    ((15, 8, 0), ("CP2", "Fz", "FC3")),
    (tuple(range(8)), LEGACY_CHANNEL_NAMES),
])
@pytest.mark.parametrize("gain", [6, 12, 24])
def test_brainflow_source_uses_board_rows_and_timestamps(monkeypatch, indices, names, gain):
    calls = []

    class FakeParams:
        ip_address = ""
        ip_port = 0
        timeout = 0

    class FakeBoardId:
        value = 77

    class FakeBoardIds:
        CYTON_DAISY_WIFI_BOARD = FakeBoardId()

    class FakeBoard:
        def __init__(self, board_id, params):
            assert board_id == 77
            assert params.ip_address == "192.168.4.1"
            self.prepared = False
            self.reads = 0

        @staticmethod
        def get_eeg_channels(board_id):
            assert board_id == 77
            return list(range(1, 17))

        @staticmethod
        def get_timestamp_channel(board_id):
            assert board_id == 77
            return 17

        @staticmethod
        def get_package_num_channel(board_id):
            return 0

        def prepare_session(self):
            self.prepared = True

        def config_board(self, command):
            calls.append(command)

        def start_stream(self, size):
            assert size == 45000

        def get_board_data(self):
            self.reads += 1
            if self.reads == 1:
                return np.empty((18, 0))
            counters = {2: [255.0, 0.0], 3: [1.0, 2.0], 4: [4.0, 5.0]}
            return np.asarray([counters[self.reads]] + [
                [10.0 * channel, 10.0 * channel + 1] for channel in range(1, 17)
            ] + [[100.000, 100.004]])

        def stop_stream(self):
            calls.append("stop")

        def is_prepared(self):
            return self.prepared

        def release_session(self):
            self.prepared = False
            calls.append("release")

    board_shim = types.ModuleType("brainflow.board_shim")
    board_shim.BoardIds = FakeBoardIds
    board_shim.BoardShim = FakeBoard
    board_shim.BrainFlowInputParams = FakeParams
    package = types.ModuleType("brainflow")
    package.board_shim = board_shim
    monkeypatch.setitem(sys.modules, "brainflow", package)
    monkeypatch.setitem(sys.modules, "brainflow.board_shim", board_shim)

    source = BrainFlowSource(
        channel_indices=indices, channel_names=names, sample_rate=250, gain=gain
    )
    with source:
        chunk = source.read()
        second = source.read()
        assert np.isclose(second.monotonic_timestamps[0] - chunk.monotonic_timestamps[-1], 0.004)
        with pytest.raises(RuntimeError, match="package discontinuity"):
            source.read()
    assert np.allclose(chunk.data, np.asarray([[10.0 * (i + 1), 10.0 * (i + 1) + 1] for i in indices]))
    assert np.allclose(np.diff(chunk.monotonic_timestamps), [0.004])
    assert chunk.received_timestamps.tolist() == [100.0, 100.004]
    assert calls[0] == "~6"
    assert calls[1] == build_all_channel_commands(indices, gain)
    assert calls[-2:] == ["stop", "release"]


def test_cli_and_live_source_defaults_enable_both_banks():
    import argparse
    from bci_maze.lk_mini.cli import add_source_arguments, source_from_args

    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    args = parser.parse_args([])
    for source in (source_from_args(args), BrainFlowSource(), LSLSource()):
        assert source.channel_indices == tuple(range(16))
        assert source.channel_names == (
            "FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4",
            "Fz", "FC1", "FCz", "FC2", "C5", "C6", "CP1", "CP2",
        )
        assert source.channel_names[:8] == LEGACY_CHANNEL_NAMES
    command = build_all_channel_commands(args.channels)
    assert command == "".join(f"x{code}060110X" for code in "12345678QWERTYUI")


def test_cli_requires_explicit_names_for_legacy_eight_channels():
    import argparse
    from bci_maze.lk_mini.cli import add_source_arguments, source_from_args

    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    args = parser.parse_args(["--channels", "1,2,3,4,5,6,7,8"])
    with pytest.raises(ValueError, match="same number"):
        source_from_args(args)
    args = parser.parse_args([
        "--channels", "1,2,3,4,5,6,7,8",
        "--channel-names", ",".join(LEGACY_CHANNEL_NAMES),
    ])
    assert source_from_args(args).channel_names == LEGACY_CHANNEL_NAMES


@pytest.mark.parametrize("stream_channels", [8, 16])
def test_lsl_reads_sixteen_channels_or_rejects_short_stream(monkeypatch, stream_channels):
    class FakeInfo:
        def nominal_srate(self):
            return 250

        def channel_count(self):
            return stream_channels

    class FakeInlet:
        def __init__(self, stream, **kwargs):
            pass

        def info(self, **kwargs):
            return FakeInfo()

        def pull_chunk(self, **kwargs):
            return [list(range(16)), list(range(100, 116))], [1.0, 1.004]

    package = types.ModuleType("pylsl")
    package.StreamInlet = FakeInlet
    package.resolve_byprop = lambda *args, **kwargs: [object()]
    monkeypatch.setitem(sys.modules, "pylsl", package)
    source = LSLSource()
    if stream_channels == 8:
        with pytest.raises(RuntimeError, match="channel 16 was requested"):
            source.start()
    else:
        with source:
            chunk = source.read()
        assert chunk.data.shape == (16, 2)
        assert chunk.data[8].tolist() == [8, 108]
        assert chunk.data[15].tolist() == [15, 115]
        assert chunk.monotonic_timestamps.tolist() == [1.0, 1.004]

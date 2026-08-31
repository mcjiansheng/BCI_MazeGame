import numpy as np
import pytest

from bci_maze.lk_mini.source import (
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


def test_synthetic_source_returns_microvolt_like_chunks():
    source = SyntheticSource(realtime=False, seed=7)
    with source:
        chunk = source.read()
    assert chunk.data.shape == (8, 25)
    assert chunk.monotonic_timestamps.shape == (25,)
    assert np.isfinite(chunk.data).all()
    assert np.diff(chunk.monotonic_timestamps).min() > 0

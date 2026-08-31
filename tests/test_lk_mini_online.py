import numpy as np
import pytest

from bci_maze.lk_mini.online import (
    ProbabilitySmoother,
    RollingEEGBuffer,
    validate_stream_metadata,
)
from bci_maze.lk_mini.source import ReplaySource


def test_rolling_buffer_keeps_latest_samples():
    buffer = RollingEEGBuffer(2, 5)
    buffer.append(np.arange(8).reshape(2, 4))
    buffer.append(np.arange(8, 14).reshape(2, 3))
    assert buffer.sample_count == 5
    assert buffer.latest(2).shape == (2, 2)
    with pytest.raises(ValueError):
        buffer.latest(6)


def test_online_helpers_reject_invalid_configuration_and_probabilities():
    with pytest.raises(ValueError, match="positive"):
        RollingEEGBuffer(2, 0)
    with pytest.raises(ValueError, match="two unique"):
        ProbabilitySmoother(("left",), history=2)
    with pytest.raises(ValueError, match="history"):
        ProbabilitySmoother(("left", "right"), history=0)
    smoother = ProbabilitySmoother(("left", "right"))
    with pytest.raises(ValueError, match="probabilities"):
        smoother.update((-0.1, 1.1))


def test_probability_smoother_warms_up_then_accepts():
    smoother = ProbabilitySmoother(("left", "right"), history=3, confidence_threshold=0.65)
    assert smoother.update((0.9, 0.1)).reason == "warming_up"
    assert smoother.update((0.8, 0.2)).reason == "warming_up"
    decision = smoother.update((0.85, 0.15))
    assert decision.accepted
    assert decision.label == "left"


def test_probability_smoother_reset_discards_stale_direction():
    smoother = ProbabilitySmoother(
        ("left", "right"), history=3, confidence_threshold=0.65
    )
    for _ in range(3):
        assert smoother.update((0.99, 0.01)).reason in {"warming_up", "accepted"}
    smoother.reset()
    assert smoother.update((0.01, 0.99)).reason == "warming_up"
    assert smoother.update((0.01, 0.99)).reason == "warming_up"
    decision = smoother.update((0.01, 0.99))
    assert decision.accepted
    assert decision.label == "right"


def test_replay_metadata_is_validated_after_source_start(tmp_path):
    channels = ("C4", "C3")
    path = tmp_path / "replay.npz"
    np.savez(
        path,
        continuous=np.zeros((2, 20)),
        sample_rate=np.asarray(500),
        channel_names=np.asarray(channels),
    )
    with ReplaySource(path, realtime=False) as source:
        with pytest.raises(ValueError, match="500 Hz"):
            validate_stream_metadata(
                source.sample_rate,
                source.channel_names,
                expected_sample_rate=250,
                expected_channel_names=("C3", "C4"),
            )
        assert validate_stream_metadata(
            source.sample_rate,
            source.channel_names,
            expected_sample_rate=500,
            expected_channel_names=channels,
        ) == (500, channels)

import numpy as np
import pytest

from bci_maze.lk_mini.online import ProbabilitySmoother, RollingEEGBuffer


def test_rolling_buffer_keeps_latest_samples():
    buffer = RollingEEGBuffer(2, 5)
    buffer.append(np.arange(8).reshape(2, 4))
    buffer.append(np.arange(8, 14).reshape(2, 3))
    assert buffer.sample_count == 5
    assert buffer.latest(2).shape == (2, 2)
    with pytest.raises(ValueError):
        buffer.latest(6)


def test_probability_smoother_warms_up_then_accepts():
    smoother = ProbabilitySmoother(("left", "right"), history=3, confidence_threshold=0.65)
    assert smoother.update((0.9, 0.1)).reason == "warming_up"
    assert smoother.update((0.8, 0.2)).reason == "warming_up"
    decision = smoother.update((0.85, 0.15))
    assert decision.accepted
    assert decision.label == "left"

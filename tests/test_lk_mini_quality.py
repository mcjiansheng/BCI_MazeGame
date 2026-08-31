import numpy as np

from bci_maze.lk_mini.quality import assess_signal_quality


def test_quality_flags_flatline_and_accepts_eeg_like_signal():
    sample_rate = 250
    time = np.arange(sample_rate * 4) / sample_rate
    good = 20 * np.sin(2 * np.pi * 10 * time) + np.random.default_rng(1).normal(0, 2, time.size)
    flat = np.zeros_like(good)
    quality = assess_signal_quality(np.vstack((good, flat)), sample_rate)
    assert quality[0].status in {"good", "warning"}
    assert quality[1].status == "bad"

import numpy as np

from bci_maze.preprocessing import FILTER_BANK_BANDS, make_filter_bank


def test_filter_bank_shape_and_finite_values():
    rng = np.random.default_rng(42)
    trials = rng.standard_normal((3, 22, 1000), dtype=np.float32)
    filtered = make_filter_bank(trials)
    assert filtered.shape == (3, len(FILTER_BANK_BANDS), 22, 1000)
    assert np.isfinite(filtered).all()


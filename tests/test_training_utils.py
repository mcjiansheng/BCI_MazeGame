"""Tests for training utilities: normalization stats, data preparation, augmentation."""

import numpy as np

from bci_maze.training import (
    _normalize_broadband,
    _normalize_filter_bank,
    apply_norm_stats,
    norm_stats_to_serializable,
    prepare_subject_data,
    repo_revision,
    segment_reconstruction,
)


def test_normalize_filter_bank_returns_stats_and_zero_mean_fit_split():
    rng = np.random.default_rng(0)
    train_x = rng.standard_normal((20, 9, 3, 100)).astype(np.float32) * 3 + 1
    test_x = rng.standard_normal((8, 9, 3, 100)).astype(np.float32) * 3 + 1
    fit_indices = np.arange(16)
    norm_train, norm_test, stats = _normalize_filter_bank(train_x, test_x, fit_indices)
    assert stats["mode"] == "filter_bank"
    assert stats["mean"].shape == (1, 9, 3, 1)
    assert stats["std"].shape == (1, 9, 3, 1)
    assert abs(norm_train[fit_indices].mean()) < 1e-4
    # Applying the frozen stats to the raw array reproduces the normalization.
    assert np.allclose(apply_norm_stats(test_x, stats), norm_test, atol=1e-6)


def test_normalize_broadband_stats_roundtrip():
    rng = np.random.default_rng(1)
    train_x = rng.standard_normal((12, 3, 200)).astype(np.float32) * 2 - 0.5
    test_x = rng.standard_normal((4, 3, 200)).astype(np.float32)
    fit_indices = np.arange(10)
    norm_train, norm_test, stats = _normalize_broadband(train_x, test_x, fit_indices)
    assert stats["mode"] == "broadband"
    assert stats["mean"].shape == (1, 3, 1)
    serializable = norm_stats_to_serializable(stats)
    assert isinstance(serializable["mean"], list)
    assert np.allclose(apply_norm_stats(train_x, serializable), norm_train, atol=1e-6)


def _write_synthetic_sessions(directory, n_trials=16, n_channels=3, n_times=100):
    rng = np.random.default_rng(2)
    for stem in ("A01T", "A01E"):
        raw = rng.standard_normal((n_trials, n_channels, n_times)).astype(np.float32)
        # Encode the trial index into the first sample so raw alignment is checkable.
        raw[:, 0, 0] = np.arange(n_trials, dtype=np.float32)
        np.savez_compressed(
            directory / f"{stem}.npz",
            x=raw,
            raw_x=raw,
            y=np.tile(np.array([0, 1]), n_trials // 2).astype(np.int64),
            artifact=np.zeros(n_trials, dtype=bool),
            channel_names=np.array(["C3", "Cz", "C4"]),
            sfreq=np.float32(250.0),
        )


def test_prepare_subject_data_exposes_norm_stats_and_raw(tmp_path):
    _write_synthetic_sessions(tmp_path)
    data = prepare_subject_data(
        tmp_path,
        subject=1,
        model_name="fbcnet",
        validation_size=0.25,
        seed=42,
        exclude_artifacts=True,
        keep_raw=True,
    )
    assert data["norm_stats"]["mode"] == "filter_bank"
    assert data["train_x"].ndim == 4  # trials x bands x channels x time
    assert len(data["train_raw_x"]) == len(data["train_y"])
    assert len(data["validation_raw_x"]) == len(data["validation_y"])
    # Raw trial indices encoded in the first sample must remain unique per split.
    encoded = data["train_raw_x"][:, 0, 0]
    assert len(set(encoded.tolist())) == encoded.size


def test_segment_reconstruction_is_deterministic_and_balanced():
    rng_seed = 7
    x = np.random.default_rng(3).standard_normal((24, 3, 64)).astype(np.float32)
    y = np.tile(np.array([0, 1, 2]), 8).astype(np.int64)
    first_x, first_y = segment_reconstruction(x, y, samples_per_class=5, rng=np.random.default_rng(rng_seed))
    second_x, second_y = segment_reconstruction(x, y, samples_per_class=5, rng=np.random.default_rng(rng_seed))
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert first_x.shape == (15, 3, 64)
    assert sorted(set(first_y.tolist())) == [0, 1, 2]
    # Every segment of every augmented trial must come from a same-class source.
    segment_size = 64 // 8
    for sample, label in zip(first_x, first_y):
        sources = x[y == label]
        for segment in range(8):
            start, stop = segment * segment_size, (segment + 1) * segment_size
            piece = sample[:, start:stop]
            assert any(np.array_equal(piece, source[:, start:stop]) for source in sources)


def test_repo_revision_returns_string():
    revision = repo_revision()
    assert isinstance(revision, str)
    assert revision != ""

"""Tests for the streaming inference module (online decoder + policy)."""

import json

import numpy as np
import pytest
import torch
from scipy.signal import sosfilt

from bci_maze.online import CausalBandpass, HysteresisPolicy, LabelMap, OnlineDecoder, load_label_map
from bci_maze.preprocessing import design_filter_bank
from bci_maze.training import apply_norm_stats


def test_hysteresis_policy_enter_exit_and_switch():
    policy = HysteresisPolicy(enter_threshold=0.6, exit_threshold=0.4)
    assert policy.update(np.array([0.5, 0.5])) is None  # nothing crosses the enter threshold
    assert policy.update(np.array([0.7, 0.3])) == 0  # enter class 0
    assert policy.update(np.array([0.5, 0.45])) == 0  # inside the hysteresis band: keep state
    assert policy.update(np.array([0.3, 0.65])) == 1  # class 1 crosses the enter threshold
    assert policy.update(np.array([0.45, 0.45])) == 1  # current class still above exit threshold
    assert policy.update(np.array([0.5, 0.3])) is None  # current class below exit threshold: release


def test_hysteresis_policy_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        HysteresisPolicy(enter_threshold=0.3, exit_threshold=0.7)
    with pytest.raises(ValueError, match="finite"):
        HysteresisPolicy().update(np.asarray([np.nan, 0.5]))


def test_label_map_validation():
    with pytest.raises(ValueError):
        LabelMap(labels=("a",), commands=("diagonal",))
    with pytest.raises(ValueError, match="unique"):
        LabelMap(labels=("a", "a"), commands=("left", "right"))
    label_map = LabelMap(labels=("left_hand", "tongue"), commands=("left", None))
    assert label_map.command_for(0) == "left"
    assert label_map.command_for(1) is None
    assert label_map.n_classes == 2


def test_load_label_map(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {"labels": ["left_hand", "right_hand"], "commands": {"left_hand": "left", "right_hand": "right"}},
        ),
        encoding="utf-8",
    )
    label_map = load_label_map(path)
    assert label_map.labels == ("left_hand", "right_hand")
    assert label_map.commands == ("left", "right")


def test_load_label_map_rejects_missing_command_entry(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps({"labels": ["left_hand", "right_hand"], "commands": {"left_hand": "left"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="right_hand"):
        load_label_map(path)


def test_causal_bandpass_matches_one_shot_causal_filter():
    rng = np.random.default_rng(0)
    n_channels, n_times = 3, 500
    trial = rng.standard_normal((n_channels, n_times)).astype(np.float32)
    bandpass = CausalBandpass(n_channels)
    streamed = bandpass.push(trial)
    reference = sosfilt(bandpass.sos, trial, axis=-1).astype(np.float32)
    assert np.allclose(streamed, reference, atol=1e-5)


def _make_fbcnet_checkpoint(tmp_path, n_channels=3, n_times=252, n_classes=2):
    from bci_maze.models import build_model

    model = build_model("fbcnet", n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    model.eval()
    rng = np.random.default_rng(1)
    fake_trials = rng.standard_normal((24, n_channels, n_times)).astype(np.float32)
    from bci_maze.preprocessing import make_filter_bank

    banked = make_filter_bank(fake_trials)
    mean = banked.mean(axis=(0, 3), keepdims=True)
    std = banked.std(axis=(0, 3), keepdims=True).clip(min=1e-6)
    checkpoint_path = tmp_path / "fbcnet.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": "fbcnet",
            "norm_stats": {
                "mode": "filter_bank",
                "mean": mean.tolist(),
                "std": std.tolist(),
                "fit_trials": 24,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path, model, {"mode": "filter_bank", "mean": mean, "std": std}


def test_online_decoder_matches_manual_window_pipeline(tmp_path):
    n_channels, n_times = 3, 252
    checkpoint_path, model, stats = _make_fbcnet_checkpoint(tmp_path, n_channels, n_times)
    label_map = LabelMap(labels=("left_hand", "right_hand"), commands=("left", "right"))
    decoder = OnlineDecoder(
        checkpoint_path,
        label_map,
        n_channels=n_channels,
        n_times=n_times,
        device="cpu",
        policy=HysteresisPolicy(enter_threshold=0.0, exit_threshold=0.0),
    )

    rng = np.random.default_rng(2)
    sos_list = design_filter_bank()
    for _ in range(4):
        trial = rng.standard_normal((n_channels, n_times)).astype(np.float32)
        decoder.reset_window()
        decoder.push_raw(trial)
        assert decoder.ready
        assert np.array_equal(decoder._buffer, trial)
        result = decoder.decode_window()

        # Manual reference: causal per-window filter bank + frozen normalization.
        # Broadcasting against the (1, 9, C, 1) statistics already yields a
        # batched (1, 9, C, T) window.
        views = np.stack(
            [sosfilt(sos, trial, axis=-1).astype(np.float32) for sos in sos_list]
        )
        window = apply_norm_stats(views, stats)
        assert window.ndim == 4
        with torch.no_grad():
            reference_logits = model(torch.from_numpy(window))
        reference_label = int(reference_logits.argmax(dim=1).item())
        reference_probabilities = torch.softmax(reference_logits, dim=1)[0].numpy()
        assert np.allclose(result["probabilities"], reference_probabilities, atol=1e-6)
        assert result["label"] == reference_label
        assert result["state"] == reference_label  # zero thresholds track argmax
        assert result["command"] == label_map.command_for(reference_label)


def test_online_decoder_requires_norm_stats(tmp_path):
    from bci_maze.models import build_model

    model = build_model("fbcnet", n_channels=3, n_times=252, n_classes=2)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "model_name": "fbcnet"}, checkpoint_path)
    label_map = LabelMap(labels=("left_hand", "right_hand"), commands=("left", "right"))
    with pytest.raises(ValueError, match="norm_stats"):
        OnlineDecoder(checkpoint_path, label_map, n_channels=3, n_times=252, device="cpu")

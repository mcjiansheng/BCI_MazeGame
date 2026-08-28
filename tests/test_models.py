import torch

from bci_maze.models import build_model


def test_all_models_forward_and_backward():
    shapes = {
        "eeg_conformer": (2, 1, 22, 1000),
        "fbcnet": (2, 9, 22, 1000),
        "se_mhaf_conformer": (2, 1, 22, 1000),
        "se_mhaf_conformer_v2": (2, 9, 22, 1000),
        "se_mhaf_conformer_v3": (2, 1, 22, 1000),
        "se_mhaf_conformer_v3_compact": (2, 1, 22, 1000),
        "se_mhaf_conformer_final": (2, 9, 22, 1000),
        "se_mhaf_conformer_final_logvar": (2, 9, 22, 1000),
    }
    for name, shape in shapes.items():
        model = build_model(name)
        output = model(torch.randn(*shape))
        assert output.shape == (2, 4)
        output.mean().backward()


def test_parameter_counts_are_nonzero_and_distinct():
    counts = {
        name: sum(parameter.numel() for parameter in build_model(name).parameters())
        for name in ("eeg_conformer", "fbcnet", "se_mhaf_conformer")
    }
    assert all(count > 0 for count in counts.values())
    assert len(set(counts.values())) == 3


def test_all_models_support_bcic2b_shapes():
    shapes = {
        "eeg_conformer": (2, 1, 3, 1000),
        "fbcnet": (2, 9, 3, 1000),
        "se_mhaf_conformer": (2, 1, 3, 1000),
        "se_mhaf_conformer_v2": (2, 9, 3, 1000),
        "se_mhaf_conformer_v3": (2, 1, 3, 1000),
        "se_mhaf_conformer_v3_compact": (2, 1, 3, 1000),
        "se_mhaf_conformer_final": (2, 9, 3, 1000),
        "se_mhaf_conformer_final_logvar": (2, 9, 3, 1000),
    }
    for name, shape in shapes.items():
        output = build_model(name, n_channels=3, n_classes=2)(torch.randn(*shape))
        assert output.shape == (2, 2)


def test_v2_starts_as_exact_fbc_residual():
    model = build_model("se_mhaf_conformer_v2")
    model.eval()
    x = torch.randn(2, 9, 22, 1000)
    with torch.no_grad():
        fbc_logits, _, scale = model.forward_branches(x)
        fused_logits = model(x)
    assert torch.equal(scale, torch.zeros_like(scale))
    assert torch.allclose(fused_logits, fbc_logits)


def test_final_model_starts_as_exact_fbc_residual():
    model = build_model("se_mhaf_conformer_final")
    model.eval()
    x = torch.randn(2, 9, 22, 1000)
    with torch.no_grad():
        fbc_logits, _, scale = model.forward_branches(x)
        fused_logits = model(x)
    assert torch.equal(scale, torch.zeros_like(scale))
    assert torch.allclose(fused_logits, fbc_logits)

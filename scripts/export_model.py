"""Export a trained checkpoint to TorchScript with a numerical parity check.

Produces a deployable artifact (e.g. for inference without the training code
path) and verifies that traced outputs match the eager model on random inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.models import build_model
from bci_maze.training import uses_filter_bank


def display_path(path: Path) -> str:
    """Show repository-relative paths when possible, absolute paths otherwise."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None, help="Defaults to outputs/exports/<checkpoint stem>.torchscript.pt")
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--n-times", type=int, default=1000)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--parity-batches", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = str(checkpoint["model_name"])
    model = build_model(
        model_name,
        n_channels=args.n_channels,
        n_times=args.n_times,
        n_classes=args.n_classes,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    bands = 9 if uses_filter_bank(model_name) else 1
    example = torch.randn(1, bands, args.n_channels, args.n_times)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    max_difference = 0.0
    with torch.no_grad():
        for _ in range(args.parity_batches):
            batch = torch.randn(2, bands, args.n_channels, args.n_times)
            difference = (model(batch) - traced(batch)).abs().max().item()
            max_difference = max(max_difference, difference)

    output = Path(args.output) if args.output else ROOT / "outputs" / "exports" / f"{checkpoint_path.stem}.torchscript.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    print(
        f"model={model_name} exported={display_path(output)} "
        f"parity_max_abs_diff={max_difference:.3e}"
    )
    return 0 if max_difference < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())

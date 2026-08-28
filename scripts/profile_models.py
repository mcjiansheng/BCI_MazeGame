"""Measure model footprint and inference/training-step costs on this workstation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.models import build_model
from bci_maze.preprocessing import _bandpass, make_filter_bank


DATASETS = {
    "bcic2a": {"n_channels": 22, "n_classes": 4},
    "bcic2b": {"n_channels": 3, "n_classes": 2},
}
MODELS = (
    "eeg_conformer",
    "fbcnet",
    "se_mhaf_conformer",
    "se_mhaf_conformer_v2",
    "se_mhaf_conformer_final",
)
BATCH_SIZES = {
    "eeg_conformer": 64,
    "fbcnet": 16,
    "se_mhaf_conformer": 64,
    "se_mhaf_conformer_v2": 16,
    "se_mhaf_conformer_final": 64,
}


def input_shape(model_name: str, channels: int, batch: int) -> tuple[int, ...]:
    return (
        (batch, 9, channels, 1000)
        if model_name in {"fbcnet", "se_mhaf_conformer_v2", "se_mhaf_conformer_final"}
        else (batch, 1, channels, 1000)
    )


@torch.inference_mode()
def gpu_latency(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> float:
    for _ in range(20):
        with torch.amp.autocast("cuda"):
            model(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        with torch.amp.autocast("cuda"):
            model(x)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


@torch.inference_mode()
def cpu_latency(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> float:
    for _ in range(10):
        model(x)
    started = time.perf_counter()
    for _ in range(repeats):
        model(x)
    return float((time.perf_counter() - started) * 1000 / repeats)


def model_variant(model_name: str, dataset_name: str) -> str:
    if model_name == "se_mhaf_conformer_final" and dataset_name == "bcic2a":
        return "se_mhaf_conformer_final_logvar"
    return model_name


def synthetic_training_peak_mb(
    model_name: str,
    build_name: str,
    channels: int,
    classes: int,
    batch_size: int,
) -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(build_name, n_channels=channels, n_classes=classes).cuda().train()
    if model_name == "se_mhaf_conformer_final":
        model.freeze_backbone()
    x = torch.randn(*input_shape(model_name, channels, batch_size), device="cuda")
    y = torch.randint(0, classes, (batch_size,), device="cuda")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        if model_name == "se_mhaf_conformer_v2":
            fbc_logits, mhaf_logits, scale = model.forward_branches(x)
            fused_logits = model.fuse_logits(fbc_logits, mhaf_logits, scale)
            loss = (
                torch.nn.functional.cross_entropy(fused_logits, y)
                + 0.25 * torch.nn.functional.cross_entropy(fbc_logits, y)
                + 0.25 * torch.nn.functional.cross_entropy(mhaf_logits, y)
            )
        elif model_name == "se_mhaf_conformer_final":
            fbc_logits, residual_logits, scale = model.forward_branches(x)
            fused_logits = model.fuse_logits(fbc_logits, residual_logits, scale)
            loss = (
                torch.nn.functional.cross_entropy(fused_logits, y)
                + 0.25 * torch.nn.functional.cross_entropy(residual_logits, y)
            )
        else:
            loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    del model, x, y, optimizer, loss
    torch.cuda.empty_cache()
    return peak


def signal_processing_cost(channels: int, repeats: int = 10) -> dict[str, float]:
    rng = np.random.default_rng(42)
    trial = rng.standard_normal((1, channels, 1000), dtype=np.float32)
    started = time.perf_counter()
    for _ in range(repeats):
        _bandpass(trial, 250.0, 4.0, 40.0, 6, 60.0)
    broadband_ms = (time.perf_counter() - started) * 1000 / repeats
    started = time.perf_counter()
    for _ in range(repeats):
        make_filter_bank(trial)
    filter_bank_ms = (time.perf_counter() - started) * 1000 / repeats
    return {
        "broadband_filter_ms_per_trial_cpu": float(broadband_ms),
        "fbcnet_filter_bank_ms_per_trial_cpu": float(filter_bank_ms),
    }


def observed_training_cost(result_paths: list[Path]) -> dict[str, object]:
    rows = []
    for result_path in result_paths:
        if result_path.exists():
            rows.extend(json.loads(result_path.read_text(encoding="utf-8"))["results"])
    output = {}
    for model_name in MODELS:
        selected = [row for row in rows if row["model"] == model_name]
        durations = np.asarray([row["duration_seconds"] for row in selected], dtype=float)
        epochs = np.asarray([row["epochs_ran"] for row in selected], dtype=float)
        checkpoint_sizes = []
        for row in selected:
            path = ROOT / row["checkpoint"]
            if path.exists():
                checkpoint_sizes.append(path.stat().st_size)
        output[model_name] = {
            "subjects": len(selected),
            "total_train_and_test_seconds": float(durations.sum()),
            "mean_seconds_per_subject": float(durations.mean()),
            "mean_epochs_ran": float(epochs.mean()),
            "weighted_seconds_per_epoch": float(durations.sum() / epochs.sum()),
            "mean_checkpoint_mb": float(np.mean(checkpoint_sizes) / (1024 ** 2)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/model_costs.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    payload: dict[str, object] = {
        "device": device_name,
        "cpu_threads_for_latency": 1,
        "notes": "GPU inference uses AMP; timings exclude disk I/O and signal filtering.",
        "datasets": {},
        "observed_training": {},
    }
    for dataset_name, spec in DATASETS.items():
        channels, classes = spec["n_channels"], spec["n_classes"]
        dataset_profiles = {"signal_processing": signal_processing_cost(channels), "models": {}}
        for model_name in MODELS:
            build_name = model_variant(model_name, dataset_name)
            cpu_model = build_model(
                build_name, n_channels=channels, n_classes=classes
            ).cpu().eval()
            cpu_x = torch.randn(*input_shape(model_name, channels, 1))
            cpu_ms = cpu_latency(cpu_model, cpu_x, repeats=50)
            profile = {
                "variant": build_name,
                "parameters": int(sum(p.numel() for p in cpu_model.parameters())),
                "fp32_parameter_mb": float(
                    sum(p.numel() * p.element_size() for p in cpu_model.parameters())
                    / (1024 ** 2)
                ),
                "cpu_batch1_ms": cpu_ms,
            }
            del cpu_model, cpu_x
            if torch.cuda.is_available():
                gpu_model = build_model(
                    build_name, n_channels=channels, n_classes=classes
                ).cuda().eval()
                for batch, repeats in ((1, 200), (64, 50)):
                    gpu_x = torch.randn(
                        *input_shape(model_name, channels, batch), device="cuda"
                    )
                    ms = gpu_latency(gpu_model, gpu_x, repeats)
                    profile[f"gpu_batch{batch}_ms"] = ms
                    profile[f"gpu_batch{batch}_trials_per_second"] = float(batch * 1000 / ms)
                    del gpu_x
                del gpu_model
                torch.cuda.empty_cache()
            profile["synthetic_training_peak_mb"] = synthetic_training_peak_mb(
                model_name,
                build_name,
                channels,
                classes,
                BATCH_SIZES[model_name],
            )
            dataset_profiles["models"][model_name] = profile
        payload["datasets"][dataset_name] = dataset_profiles
        result_paths = [
            ROOT / "outputs" / f"{dataset_name}_benchmark.json",
            ROOT / "outputs" / f"{dataset_name}_se_mhaf_v2.json",
            ROOT / "outputs" / f"{dataset_name}_se_mhaf_final.json",
        ]
        payload["observed_training"][dataset_name] = observed_training_cost(result_paths)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

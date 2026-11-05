"""Train and validate one participant's LK-Mini CSP+LDA MI model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bci_maze.lk_mini.model import MIModelConfig, save_model_bundle, train_subject_model
from bci_maze.lk_mini.provenance import MODEL_DATA_ORIGINS, recording_source, require_motor_imagery
from bci_maze.lk_mini.recording import load_mi_recordings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--low-freq", type=float, default=7.0)
    parser.add_argument("--high-freq", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=4)
    parser.add_argument("--include-artifacts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loaded = load_mi_recordings(args.recordings)
    recordings = loaded["metadata"]
    recorded_subjects = {str(metadata.get("subject_id")) for metadata in recordings}
    if recorded_subjects != {args.subject}:
        raise ValueError(
            f"Recording subject IDs {sorted(recorded_subjects)} do not match "
            f"--subject {args.subject!r}"
        )
    sources = set()
    for path, metadata in zip(args.recordings, recordings):
        try:
            source = recording_source(metadata)
            require_motor_imagery(metadata)
        except ValueError as error:
            raise ValueError(f"Recording {path}: {error}") from error
        if source not in MODEL_DATA_ORIGINS:
            raise ValueError(f"Recording {path} has unsupported source for personal training: {source!r}")
        sources.add(source)
    if len(sources) != 1:
        raise ValueError("Personal training requires a single data origin; keep synthetic and recorded EEG separate")
    source = sources.pop()
    keep = ~loaded["artifact"] if not args.include_artifacts else slice(None)
    epochs = loaded["epochs"][keep]
    labels = loaded["labels"][keep]
    blocks = loaded["blocks"][keep]
    sample_rate = int(loaded["sample_rate"].item())
    channel_names = tuple(loaded["channel_names"].tolist())
    print(f"训练输入：{len(epochs)} trials × {len(channel_names)} 通道，{sample_rate} Hz")
    epoch_seconds = epochs.shape[-1] / sample_rate
    config = MIModelConfig(
        sample_rate=sample_rate,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        csp_components=args.csp_components,
        epoch_seconds=epoch_seconds,
    )
    bundle, report = train_subject_model(
        epochs,
        labels,
        blocks,
        channel_names=channel_names,
        subject_id=args.subject,
        training_files=[str(path) for path in args.recordings],
        config=config,
        seed=args.seed,
    )
    recorded = source in {"hardware", "lsl"}
    provenance = {
        "data_origin": MODEL_DATA_ORIGINS[source],
        "task": "motor_imagery",
        "calibration_passed": bool(recorded and report["above_chance_95_percent"]),
        "human_mi_verified": recorded and all(
            metadata.get("provenance", {}).get("human_mi_verified") is True
            for metadata in recordings
        ),
    }
    bundle.metadata["provenance"] = provenance
    report.update(provenance)
    output = args.output or Path("outputs/subject_models") / args.subject / "csp_lda.joblib"
    destination = save_model_bundle(bundle, report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"模型：{destination}\n报告：{destination.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

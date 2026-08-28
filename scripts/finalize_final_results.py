from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Add explicit final-model variant metadata")
    parser.add_argument("--result", required=True)
    parser.add_argument("--logvar-only", action="store_true")
    args = parser.parse_args()
    path = Path(args.result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    variant = (
        "se_mhaf_conformer_final_logvar"
        if args.logvar_only
        else "se_mhaf_conformer_final"
    )
    for row in payload["results"]:
        row["model_variant"] = variant
        row["logvar_only"] = args.logvar_only
        checkpoint_path = Path(row["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["model_name"] = variant
        checkpoint["logvar_only"] = args.logvar_only
        torch.save(checkpoint, checkpoint_path)
        row["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

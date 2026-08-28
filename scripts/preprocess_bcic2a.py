from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.preprocessing import preprocess_dataset


def parse_subjects(value: str) -> list[int]:
    if value.lower() == "all":
        return list(range(1, 10))
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV 2a GDF files")
    parser.add_argument("--raw-dir", default="data/BCIC2a")
    parser.add_argument("--label-dir", default="data/BCIC2a/true_labels")
    parser.add_argument("--output-dir", default="data/processed/bcic2a")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = preprocess_dataset(
        args.raw_dir,
        args.label_dir,
        args.output_dir,
        parse_subjects(args.subjects),
        overwrite=args.overwrite,
    )
    print(f"Prepared {len(paths)} sessions in {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

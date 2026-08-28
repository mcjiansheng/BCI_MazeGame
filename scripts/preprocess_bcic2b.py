from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.preprocessing_bcic2b import preprocess_bcic2b


def parse_subjects(value: str) -> list[int]:
    if value.lower() == "all":
        return list(range(1, 10))
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess official BNCI 004-2014 MAT files")
    parser.add_argument("--raw-dir", default="data/BCIC2b")
    parser.add_argument("--output-dir", default="data/processed/bcic2b")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = preprocess_bcic2b(
        args.raw_dir,
        args.output_dir,
        parse_subjects(args.subjects),
        overwrite=args.overwrite,
    )
    print(f"Prepared {len(paths)} subject splits in {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()


"""Run acquisition and CSP or SE-MHAF workflows with their local runtimes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "ui": "bci_dashboard.py",
    "check": "lk_mini_check.py",
    "collect": "collect_lk_mi.py",
    "train": "train_lk_mi.py",
    "pretrain": "pretrain_lk_semhaf.py",
    "finetune": "finetune_lk_semhaf.py",
    "online": "online_lk_mi.py",
    "validate": "validate_lk_mini_workflow.py",
    "demo": "lk_mini_check.py",
}


def requires_torch(command: str, arguments: list[str]) -> bool:
    if command in {"pretrain", "finetune"}:
        return True
    if command != "online":
        return False
    for index, argument in enumerate(arguments):
        if argument.startswith("--model="):
            return Path(argument.split("=", 1)[1]).suffix.lower() == ".pt"
        if argument == "--model" and index + 1 < len(arguments):
            return Path(arguments[index + 1]).suffix.lower() == ".pt"
    return False


def workflow_python(command: str, arguments: list[str]) -> str:
    if not requires_torch(command, arguments):
        return sys.executable
    candidate = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"SE-MHAF requires the PyTorch environment at {candidate}. See doc/SE_MHAF_LIVE.md"
        )
    return str(candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(SCRIPTS))
    args, extra = parser.parse_known_args()
    if args.command == "ui":
        # Keep the server in this process so Ctrl+C reaches its owned-task cleanup.
        sys.path.insert(0, str(ROOT / "src"))
        from bci_maze.dashboard.server import main as dashboard_main
        return dashboard_main(extra)
    if args.command == "demo":
        extra = ["--backend", "synthetic", "--duration", "20", *extra]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.call(
        [workflow_python(args.command, extra), "-X", "utf8", str(ROOT / "scripts" / SCRIPTS[args.command]), *extra],
        cwd=ROOT, env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())

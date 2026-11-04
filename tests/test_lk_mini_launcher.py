import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command,arguments,expected", [
    ("pretrain", [], True),
    ("finetune", [], True),
    ("online", ["--model", "models/person.PT"], True),
    ("online", ["--model=models/person.pt"], True),
    ("online", ["--model", "models/person.joblib"], False),
    ("collect", ["--preset", "short"], False),
])
def test_launcher_selects_deep_runtime(command, arguments, expected):
    from scripts.run_lk_mi import requires_torch
    assert requires_torch(command, arguments) is expected


def test_python_launcher_propagates_failure():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_lk_mi.py"), "check", "--duration", "-1"],
        cwd=ROOT, capture_output=True, timeout=20,
    )
    assert result.returncode != 0


@pytest.mark.skipif(os.name != "nt" or not (ROOT / ".venv-mi").exists(), reason="local Windows launcher")
def test_windows_launcher_propagates_failure():
    result = subprocess.run(
        ["cmd.exe", "/c", str(ROOT / "bci-mi.cmd"), "check", "--duration", "-1"],
        cwd=ROOT, capture_output=True, timeout=20,
    )
    assert result.returncode != 0

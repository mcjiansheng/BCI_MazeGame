"""Launch the local MindMaze dashboard from the project checkout."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.dashboard.server import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the Phase 1 manual visualiser without requiring editable installation first."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualiser.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["manual-visualise", *sys.argv[1:]]))

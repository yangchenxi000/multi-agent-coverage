"""Run all experiments and build report."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "scripts/run_smoke_single.py"])
    run([py, "scripts/run_single_obstacle.py"])
    run([py, "scripts/run_two_agents.py"])
    run([py, "scripts/run_three_agents.py"])
    run([py, "scripts/run_ablation.py"])
    run([py, "scripts/make_report.py"])


if __name__ == "__main__":
    main()

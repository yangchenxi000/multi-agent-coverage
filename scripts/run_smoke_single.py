"""Run Experiment 0: single-agent smoke test from straight-line trajectory."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.experiment_runner import run_single_variant
from src.utils import ensure_dir, save_yaml, set_seed


def main() -> None:
    cfg_path = ROOT / "configs" / "single_agent.yaml"
    out_dir = ROOT / "outputs" / "smoke_single"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    ensure_dir(out_dir)
    save_yaml(out_dir / "config_used.yaml", cfg)

    metrics = run_single_variant(
        cfg,
        out_dir,
        "smoke_single",
        use_coupling=False,
        use_obstacle_mask=False,
    )
    if (out_dir / "trajectory_evolution.gif").exists():
        shutil.copy2(out_dir / "trajectory_evolution.gif", out_dir / "evolution.gif")

    (out_dir / "short_report.md").write_text(
        "# Smoke Single\n\n"
        "- Init is a straight trajectory (not random particles).\n"
        "- Saved snapshots: iter_000, iter_010, iter_050, iter_100.\n"
        f"- Final ergodic metric: {metrics['ergodic_metric']:.6f}\n"
        f"- Final obstacle violation ratio: {metrics['obstacle_violation_ratio']:.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

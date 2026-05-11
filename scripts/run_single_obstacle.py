"""Run Experiment 1: single-agent obstacle-aware density comparison."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.experiment_runner import run_single_variant
from src.utils import ensure_dir, save_yaml, set_seed


def main() -> None:
    cfg_base = yaml.safe_load((ROOT / "configs" / "single_agent.yaml").read_text(encoding="utf-8"))
    # no_obstacle_aware: paper-friendly (no obstacle).
    cfg_no = dict(cfg_base)
    cfg_no["scenario"] = dict(cfg_base["scenario"])
    cfg_no["scenario"]["name"] = "paper_no_obstacle_scene"
    cfg_no["scenario"]["hotspots"] = [
        {"mean": [0.34, 0.50], "cov": [[0.012, 0.0], [0.0, 0.01]], "weight": 0.33},
        {"mean": [0.64, 0.58], "cov": [[0.012, 0.0], [0.0, 0.018]], "weight": 0.34},
        {"mean": [0.53, 0.25], "cov": [[0.02, 0.0], [0.0, 0.01]], "weight": 0.33},
    ]
    cfg_no["scenario"]["obstacles"] = []

    # with_obstacle_aware: restore original obstacle scene.
    cfg_yes = dict(cfg_base)
    cfg_yes["scenario"] = dict(cfg_base["scenario"])
    cfg_yes["scenario"]["name"] = "user_custom_scene"
    cfg_yes["scenario"]["hotspots"] = []
    cfg_yes["scenario"]["obstacles"] = []

    for c in (cfg_no, cfg_yes):
        c["optimization"]["iterations"] = 220
        c["optimization"]["snapshot_iters"] = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
        c["stein"]["alpha"] = 0.028

    set_seed(int(cfg_base["seed"]))

    out_root = ROOT / "outputs" / "single_obstacle"
    ensure_dir(out_root)

    m_no = run_single_variant(cfg_no, out_root / "no_obstacle_aware", "no_obstacle_aware", use_coupling=False, use_obstacle_mask=False)
    m_yes = run_single_variant(cfg_yes, out_root / "with_obstacle_aware", "with_obstacle_aware", use_coupling=False, use_obstacle_mask=True)

    df = pd.DataFrame([m_no, m_yes])
    df.to_csv(out_root / "metrics.csv", index=False)
    df.to_json(out_root / "metrics.json", orient="records", indent=2)
    df.to_csv(out_root / "single_obstacle_comparison.csv", index=False)
    save_yaml(out_root / "config_used.yaml", {"no_obstacle_aware": cfg_no, "with_obstacle_aware": cfg_yes})

    t_no = np.load(out_root / "no_obstacle_aware" / "trajectories.npy")
    t_yes = np.load(out_root / "with_obstacle_aware" / "trajectories.npy")
    c_no = np.load(out_root / "no_obstacle_aware" / "controls.npy")
    c_yes = np.load(out_root / "with_obstacle_aware" / "controls.npy")
    np.save(out_root / "trajectories.npy", np.stack([t_no, t_yes], axis=0))
    np.save(out_root / "controls.npy", np.stack([c_no, c_yes], axis=0))

    (out_root / "short_report.md").write_text(
        "# Single Agent (No Obstacle, Paper Figure)\n\n"
        "Compare no obstacle-aware density vs with obstacle-aware density under no-obstacle scene.\n"
        f"- no obstacle-aware min distance: {m_no['min_obstacle_distance']:.6f}\n"
        f"- with obstacle-aware min distance: {m_yes['min_obstacle_distance']:.6f}\n"
        f"- no obstacle-aware violation ratio: {m_no['obstacle_violation_ratio']:.6f}\n"
        f"- with obstacle-aware violation ratio: {m_yes['obstacle_violation_ratio']:.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

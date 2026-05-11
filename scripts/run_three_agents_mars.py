"""Run three-agent baseline vs ours on auto-generated Mars DEM scene."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.experiment_runner import run_compare_two_configs
from src.utils import save_yaml, set_seed


def _resolve_dem_path(default_rel_path: str) -> str:
    """Resolve DEM path from repo root first, then parent folder as fallback."""
    p_root = (ROOT / default_rel_path).resolve()
    if p_root.exists():
        return str(p_root)
    p_parent = (ROOT.parent / default_rel_path).resolve()
    if p_parent.exists():
        return str(p_parent)
    return str(p_root)


def main() -> None:
    baseline_cfg = yaml.safe_load((ROOT / "configs" / "three_agents_mars_baseline.yaml").read_text(encoding="utf-8"))
    ours_cfg = yaml.safe_load((ROOT / "configs" / "three_agents_mars_ours.yaml").read_text(encoding="utf-8"))
    dem_path = _resolve_dem_path(str(baseline_cfg["mars"]["dem_path"]))
    baseline_cfg["mars"]["dem_path"] = dem_path
    ours_cfg["mars"]["dem_path"] = dem_path

    set_seed(int(ours_cfg["seed"]))
    out_dir = ROOT / "outputs" / "three_agents_mars"
    run_compare_two_configs(baseline_cfg, ours_cfg, out_dir, comparison_name="baseline_vs_ours_3agents_mars")

    save_yaml(out_dir / "config_used.yaml", {"baseline": baseline_cfg, "ours": ours_cfg})
    shutil.copy2(out_dir / "baseline" / "ergodic_curve.png", out_dir / "ergodic_metric_curve.png")
    shutil.copy2(out_dir / "baseline" / "overlap_curve.png", out_dir / "overlap_curve_baseline.png")
    shutil.copy2(out_dir / "ours" / "overlap_curve.png", out_dir / "overlap_curve_ours.png")
    tb = np.load(out_dir / "baseline" / "trajectories.npy")
    to = np.load(out_dir / "ours" / "trajectories.npy")
    cb = np.load(out_dir / "baseline" / "controls.npy")
    co = np.load(out_dir / "ours" / "controls.npy")
    np.save(out_dir / "trajectories.npy", np.stack([tb, to], axis=0))
    np.save(out_dir / "controls.npy", np.stack([cb, co], axis=0))


if __name__ == "__main__":
    main()

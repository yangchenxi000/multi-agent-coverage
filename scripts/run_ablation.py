"""Run Experiment 4: ablation study."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.experiment_runner import run_ablation
from src.utils import save_json, save_yaml, set_seed


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "ablation.yaml").read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    out_dir = ROOT / "outputs" / "ablation"
    df = run_ablation(cfg, out_dir)
    df.to_csv(out_dir / "metrics.csv", index=False)
    save_json(out_dir / "metrics.json", df.to_dict(orient="records"))
    save_yaml(out_dir / "config_used.yaml", cfg)

    traj_list = []
    ctrl_list = []
    for case in df["ablation_case"].tolist():
        traj_p = out_dir / case / "trajectories.npy"
        ctrl_p = out_dir / case / "controls.npy"
        if traj_p.exists() and ctrl_p.exists():
            traj_list.append(np.load(traj_p))
            ctrl_list.append(np.load(ctrl_p))
    if traj_list and ctrl_list:
        np.save(out_dir / "trajectories.npy", np.stack(traj_list, axis=0))
        np.save(out_dir / "controls.npy", np.stack(ctrl_list, axis=0))

    (out_dir / "ablation_report.md").write_text(
        "# Ablation\n\n"
        "Included: no coupling, no obstacle-aware density, gamma/alpha/kernel/step/smoothness sweeps.\n"
        "See `ablation_results.csv` and generated plots for details.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

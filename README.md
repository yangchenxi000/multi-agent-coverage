# Project Code (Reproducible Package)

This folder contains the runnable code package for reproducing the trajectory-based Stein coverage experiments.

## 1. What Is Implemented

Core method: multi-agent trajectory optimization with Stein-style update terms.

- Task term: drives trajectories to match target coverage density.
- Obstacle-aware term: reduces probability mass in non-traversable regions via SDF-based mask.
- Self-repulsion term: separates trajectories of agents to reduce overlap.
- Coupling term: coordinated update among agents.

Trajectory-first design:

- Initialize each agent with straight-line trajectory controls.
- Treat trajectory points as optimization objects (not random SVGD particles).
- Update full trajectories over iterations.

## 2. Folder Layout

- `src/`: method implementation (map/SDF/density/Stein/update/metrics/runner)
- `configs/`: all experiment configs
- `scripts/`: experiment entrypoints
- `tests/`: unit tests
- `docs/`: notes/formulas/deviations
- `outputs/`: generated during runs (ignored by git)

## 3. Environment Setup

```bash
python -m pip install -r requirements.txt
```

## 4. Quick Verification (Recommended)

Run tests:

```bash
pytest -q
```

Run a minimal experiment:

```bash
python scripts/run_smoke_single.py
```

Expected artifacts under `outputs/smoke_single/` include:

- `metrics.json`
- `trajectory_overlay.png`
- `trajectory_evolution.gif`

## 5. Main Usage

Baseline vs ours (2 agents):

```bash
python scripts/run_two_agents.py
```

Baseline vs ours (3 agents):

```bash
python scripts/run_three_agents.py
```

Ablation suite:

```bash
python scripts/run_ablation.py
```

Run all and build report:

```bash
python scripts/run_all.py
```

## 6. Mars DEM Dependency

Mars scenarios require this file at this folder root:

- `Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`

It is intentionally excluded from git because of file size.






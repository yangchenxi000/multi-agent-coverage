"""Experiment orchestration and artifact generation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .density_utils import (
    build_density_field,
    gaussian_mixture_density,
    obstacle_aware_density,
    query_density_field,
)
from .map_utils import Workspace, build_obstacle_mask, default_scenarios
from .mars_scene import build_mars_scene_from_dem
from .optimizer import (
    optimize_trajectories_approx,
    optimize_trajectories_line_svgd,
    optimize_trajectories_lqfm_fallback,
)
from .sdf_utils import compute_sdf, sdf_gradient, traversability_mask_from_sdf
from .utils import Timer, ensure_dir, save_json, save_yaml, set_seed
from .viz import (
    make_trajectory_gif,
    plot_combined_map,
    plot_comparison_panel,
    plot_curves,
    plot_heatmap,
    plot_trajectory_point_vectors,
    plot_trajectories,
    plot_vector_field,
)


def _infer_agent_goals(cfg: Dict, n_agents: int, initial_states: np.ndarray) -> np.ndarray:
    """Infer per-agent terminal goals for straight-line initialization."""
    if "goal_positions" in cfg.get("agents", {}):
        g = np.asarray(cfg["agents"]["goal_positions"], dtype=float)
        if g.shape == (n_agents, 2):
            return g

    hotspots = cfg.get("scenario", {}).get("hotspots", [])
    if hotspots:
        # Assign agents to strongest hotspots cyclically.
        hs = sorted(hotspots, key=lambda z: float(z.get("weight", 1.0)), reverse=True)
        goals = np.array([hs[i % len(hs)]["mean"] for i in range(n_agents)], dtype=float)
        return goals

    # Fallback: keep direction towards workspace center.
    center = np.array(
        [
            0.5 * sum(cfg["workspace"]["xlim"]),
            0.5 * sum(cfg["workspace"]["ylim"]),
        ],
        dtype=float,
    )
    return np.repeat(center[None, :], n_agents, axis=0)


def _default_controls(cfg: Dict, initial_states: np.ndarray, horizon: int) -> np.ndarray:
    """Build straight-line trajectory tracking controls as Stein initialization."""
    n_agents = initial_states.shape[0]
    dt = float(cfg["sim"]["dt"])
    max_acc = float(cfg["sim"]["max_accel"])
    goals = _infer_agent_goals(cfg, n_agents, initial_states)
    controls = np.zeros((n_agents, horizon, 2), dtype=float)

    init_kp = float(cfg.get("optimization", {}).get("init_line_kp", 6.0))
    init_kv = float(cfg.get("optimization", {}).get("init_line_kv", 2.2))

    for i in range(n_agents):
        p0 = initial_states[i, :2].copy()
        v0 = initial_states[i, 2:4].copy()
        goal = goals[i]
        v_ref = (goal - p0) / max(horizon * dt, 1e-8)
        p = p0.copy()
        v = v0.copy()
        for t in range(horizon):
            # Straight-line reference position from start to goal.
            s = (t + 1) / horizon
            p_ref = (1.0 - s) * p0 + s * goal
            a = init_kp * (p_ref - p) + init_kv * (v_ref - v)
            n = np.linalg.norm(a)
            if n > max_acc:
                a = a * (max_acc / n)
            controls[i, t] = a
            p = p + v * dt
            v = v + a * dt
    return controls


def _interp_term_field(
    X: np.ndarray,
    Y: np.ndarray,
    trajectories: np.ndarray,
    terms: np.ndarray,
    bw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """RBF interpolation from trajectory points to grid for field visualization."""
    pts = trajectories.reshape(-1, 2)
    vec = terms.reshape(-1, 2)
    grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
    d2 = np.sum((grid[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    w = np.exp(-d2 / max(bw, 1e-8))
    wn = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-10)
    out = wn @ vec
    U = out[:, 0].reshape(X.shape)
    V = out[:, 1].reshape(X.shape)
    return U, V


def _prepare_environment(cfg: Dict, output_dir: Path):
    workspace = Workspace(
        xlim=tuple(cfg["workspace"]["xlim"]),
        ylim=tuple(cfg["workspace"]["ylim"]),
        resolution=int(cfg["workspace"]["resolution"]),
    )

    if cfg["scenario"]["name"] == "mars_auto":
        mars_cfg = cfg.get("mars", {})
        dem_path = mars_cfg.get("dem_path", "")
        if not dem_path:
            raise ValueError("mars.dem_path must be set for scenario 'mars_auto'")
        mars_data = build_mars_scene_from_dem(
            dem_path=dem_path,
            workspace=workspace,
            crop_size=int(mars_cfg.get("crop_size", 720)),
            slope_percentile=float(mars_cfg.get("slope_percentile", 88.0)),
            obstacle_dilation=int(mars_cfg.get("obstacle_dilation", 1)),
            complexity=str(mars_cfg.get("complexity", "medium")),
            crop_y0=mars_cfg.get("crop_y0", None),
            crop_x0=mars_cfg.get("crop_x0", None),
            obstacle_opening=int(mars_cfg.get("obstacle_opening", 0)),
            min_component_pixels=int(mars_cfg.get("min_component_pixels", 0)),
            density_smooth_sigma=float(mars_cfg.get("density_smooth_sigma", 2.0)),
            hotspot_count=int(mars_cfg.get("hotspot_count", 2)),
            hotspot_sigma=float(mars_cfg.get("hotspot_sigma", 0.08)),
            hotspot_blend=float(mars_cfg.get("hotspot_blend", 0.45)),
        )
        X, Y = workspace.meshgrid()
        obstacle_mask = mars_data.obstacle_mask
        base_field = build_density_field(mars_data.base_density, workspace)
        sdf = compute_sdf(obstacle_mask, workspace)
        mask = traversability_mask_from_sdf(sdf, alpha=float(cfg["stein"]["alpha"]))
        aware_field = obstacle_aware_density(base_field.q, mask, workspace)

        plot_heatmap(base_field.q, workspace, "Task Density", output_dir / "task_density.png")
        plot_heatmap(base_field.q, workspace, "Raw Target Density", output_dir / "raw_target_density.png")
        plot_heatmap(obstacle_mask.astype(float), workspace, "Obstacle Map", output_dir / "obstacle_map.png", cmap="gray_r")
        plot_combined_map(base_field.q, obstacle_mask, workspace, output_dir / "combined_map.png")
        plot_heatmap(sdf, workspace, "SDF Heatmap", output_dir / "sdf_heatmap.png", cmap="coolwarm")
        plot_heatmap(mask, workspace, "Traversability Mask", output_dir / "traversability_mask.png")
        plot_heatmap(aware_field.q, workspace, "Obstacle-aware Density", output_dir / "obstacle_aware_density.png")
        plot_heatmap(mars_data.dem_patch, workspace, "Mars DEM Patch", output_dir / "mars_dem_patch.png", cmap="terrain")

        gx, gy = sdf_gradient(sdf, workspace)
        plot_vector_field(X, Y, gx, gy, workspace, "SDF Gradient Field", output_dir / "grad_field.png")
        save_json(output_dir / "mars_scene_metadata.json", mars_data.metadata)
        return workspace, X, Y, obstacle_mask, sdf, base_field, aware_field

    scenarios = default_scenarios()
    if cfg["scenario"]["name"] in scenarios:
        scn = scenarios[cfg["scenario"]["name"]]
    else:
        scn = {
            "hotspots": cfg["scenario"]["hotspots"],
            "obstacles": cfg["scenario"]["obstacles"],
        }

    X, Y, obstacle_mask = build_obstacle_mask(workspace, scn["obstacles"])
    base_q = gaussian_mixture_density(X, Y, scn["hotspots"])
    base_field = build_density_field(base_q, workspace)

    sdf = compute_sdf(obstacle_mask, workspace)
    mask = traversability_mask_from_sdf(sdf, alpha=float(cfg["stein"]["alpha"]))
    aware_field = obstacle_aware_density(base_field.q, mask, workspace)

    plot_heatmap(base_field.q, workspace, "Task Density", output_dir / "task_density.png")
    plot_heatmap(base_field.q, workspace, "Raw Target Density", output_dir / "raw_target_density.png")
    plot_heatmap(obstacle_mask.astype(float), workspace, "Obstacle Map", output_dir / "obstacle_map.png", cmap="gray_r")
    plot_combined_map(base_field.q, obstacle_mask, workspace, output_dir / "combined_map.png")
    plot_heatmap(sdf, workspace, "SDF Heatmap", output_dir / "sdf_heatmap.png", cmap="coolwarm")
    plot_heatmap(mask, workspace, "Traversability Mask", output_dir / "traversability_mask.png")
    plot_heatmap(aware_field.q, workspace, "Obstacle-aware Density", output_dir / "obstacle_aware_density.png")
    plot_heatmap(aware_field.q, workspace, "Obstacle-aware Density", output_dir / "target_density_obstacle_aware.png")

    gx, gy = sdf_gradient(sdf, workspace)
    plot_vector_field(X, Y, gx, gy, workspace, "SDF Gradient Field", output_dir / "grad_field.png")

    return workspace, X, Y, obstacle_mask, sdf, base_field, aware_field


def run_single_variant(
    cfg: Dict,
    output_dir: Path,
    variant_name: str,
    use_coupling: bool,
    use_obstacle_mask: bool,
) -> Dict[str, float]:
    """Run one variant (baseline or ours) and save artifacts."""
    ensure_dir(output_dir)
    save_yaml(output_dir / "config_used.yaml", cfg)

    workspace, X, Y, obstacle_mask, sdf, base_field, aware_field = _prepare_environment(cfg, output_dir)
    active_field = aware_field if use_obstacle_mask else base_field

    n_agents = int(cfg["agents"]["num_agents"])
    horizon = int(cfg["sim"]["horizon"])
    initial_states = np.array(cfg["agents"]["initial_states"], dtype=float)
    controls_init = _default_controls(cfg, initial_states, horizon)

    timer = Timer()
    mode = cfg["optimization"].get("optimizer_mode", "approx")
    if mode == "lqfm":
        res = optimize_trajectories_lqfm_fallback(
            initial_states,
            controls_init,
            workspace,
            active_field,
            sdf,
            cfg,
            use_coupling,
            use_obstacle_mask,
        )
    elif mode == "line_svgd":
        res = optimize_trajectories_line_svgd(
            initial_states,
            controls_init,
            workspace,
            active_field,
            sdf,
            cfg,
            use_coupling,
            use_obstacle_mask,
        )
    else:
        res = optimize_trajectories_approx(
            initial_states,
            controls_init,
            workspace,
            active_field,
            sdf,
            cfg,
            use_coupling,
            use_obstacle_mask,
        )
    runtime = timer.elapsed()

    metrics = dict(res.metrics_last)
    metrics["runtime_sec"] = runtime
    metrics["variant"] = variant_name
    metrics["use_coupling"] = use_coupling
    metrics["use_obstacle_mask"] = use_obstacle_mask

    np.save(output_dir / "trajectories.npy", res.trajectories)
    np.save(output_dir / "controls.npy", res.controls)
    np.save(output_dir / "states.npy", res.states)

    save_json(output_dir / "metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    plot_trajectories(
        res.trajectories,
        workspace,
        output_dir / "trajectory_overlay.png",
        density=active_field.q,
        obstacle_mask=obstacle_mask,
        title=f"{variant_name} trajectory",
    )
    if 0 in res.snapshots:
        plot_trajectories(
            res.snapshots[0],
            workspace,
            output_dir / "initial_trajectories.png",
            density=active_field.q,
            obstacle_mask=obstacle_mask,
            title=f"{variant_name} initial trajectories",
        )

    curve_fig = {
        "ergodic": res.metric_curve["ergodic_metric"],
        "overlap": res.metric_curve["overlap_score"],
        "violation": res.metric_curve["obstacle_violation_ratio"],
    }
    plot_curves({"ergodic_metric": curve_fig["ergodic"]}, "value", "Ergodic Metric vs Iteration", output_dir / "ergodic_curve.png")
    plot_curves({"ergodic_metric": curve_fig["ergodic"]}, "value", "Ergodic Metric vs Iteration", output_dir / "ergodic_metric_curve.png")
    plot_curves({"overlap": curve_fig["overlap"]}, "value", "Overlap vs Iteration", output_dir / "overlap_curve.png")

    last_terms = res.stein_terms_last
    total_task = np.stack([t.task_term for t in last_terms], axis=0)
    total_self = np.stack([t.self_repulsion_term for t in last_terms], axis=0)
    total_cpl = np.stack([t.coupling_term for t in last_terms], axis=0)
    total_all = np.stack([t.total for t in last_terms], axis=0)

    bw = float(cfg["stein"]["self_bandwidth"])
    U_task, V_task = _interp_term_field(X, Y, res.trajectories, total_task, bw)
    U_self, V_self = _interp_term_field(X, Y, res.trajectories, total_self, bw)
    U_cpl, V_cpl = _interp_term_field(X, Y, res.trajectories, total_cpl, max(bw, float(cfg["stein"]["coupling_bandwidth"])))
    U_tot, V_tot = _interp_term_field(X, Y, res.trajectories, total_all, bw)

    plot_vector_field(X, Y, U_task, V_task, workspace, "Task Term Field", output_dir / "task_term_field.png", background=active_field.q)
    plot_vector_field(X, Y, U_self, V_self, workspace, "Self Repulsion Field", output_dir / "self_repulsion_field.png", background=active_field.q)
    plot_vector_field(X, Y, U_cpl, V_cpl, workspace, "Coupling Field", output_dir / "coupling_field.png", background=active_field.q)
    plot_vector_field(X, Y, U_tot, V_tot, workspace, "Total Reference Flow", output_dir / "total_reference_flow.png", background=active_field.q)
    plot_trajectory_point_vectors(
        res.trajectories,
        total_task,
        workspace,
        output_dir / "task_term_on_trajectory_points.png",
        title="Task Term on Trajectory Points",
        density=active_field.q,
        obstacle_mask=obstacle_mask,
    )
    plot_trajectory_point_vectors(
        res.trajectories,
        total_self,
        workspace,
        output_dir / "self_repulsion_on_trajectory_points.png",
        title="Self Repulsion on Trajectory Points",
        density=active_field.q,
        obstacle_mask=obstacle_mask,
    )
    plot_trajectory_point_vectors(
        res.trajectories,
        total_cpl,
        workspace,
        output_dir / "coupling_on_trajectory_points.png",
        title="Coupling on Trajectory Points",
        density=active_field.q,
        obstacle_mask=obstacle_mask,
    )
    plot_trajectory_point_vectors(
        res.trajectories,
        total_all,
        workspace,
        output_dir / "total_stein_direction_on_trajectory_points.png",
        title="Total Stein Direction on Trajectory Points",
        density=active_field.q,
        obstacle_mask=obstacle_mask,
    )

    sorted_snaps = [res.snapshots[k] for k in sorted(res.snapshots)]
    if sorted_snaps:
        make_trajectory_gif(sorted_snaps, workspace, output_dir / "trajectory_evolution.gif", density=active_field.q)
    for k, tr in sorted(res.snapshots.items()):
        plot_trajectories(
            tr,
            workspace,
            output_dir / f"iter_{k:03d}.png",
            density=active_field.q,
            obstacle_mask=obstacle_mask,
            title=f"{variant_name} iter {k}",
        )

    note = (
        f"# {variant_name}\n\n"
        f"- use_obstacle_mask: {use_obstacle_mask}\n"
        f"- use_coupling: {use_coupling}\n"
        f"- final ergodic metric: {metrics['ergodic_metric']:.6f}\n"
        f"- overlap score: {metrics['overlap_score']:.6f}\n"
        f"- obstacle violation ratio: {metrics['obstacle_violation_ratio']:.6f}\n"
    )
    (output_dir / "experiment_note.md").write_text(note, encoding="utf-8")
    return metrics


def run_baseline_vs_ours(cfg: Dict, output_dir: Path) -> pd.DataFrame:
    """Run baseline and ours variants and save comparison figures/table."""
    ensure_dir(output_dir)
    save_yaml(output_dir / "config_used.yaml", cfg)

    base_cfg = copy.deepcopy(cfg)
    ours_cfg = copy.deepcopy(cfg)

    metrics_base = run_single_variant(
        base_cfg,
        output_dir / "baseline",
        "baseline",
        use_coupling=False,
        use_obstacle_mask=False if cfg.get("force_no_obstacle", False) else cfg["stein"].get("baseline_use_obstacle_mask", False),
    )
    metrics_ours = run_single_variant(
        ours_cfg,
        output_dir / "ours",
        "ours",
        use_coupling=True,
        use_obstacle_mask=True,
    )

    df = pd.DataFrame([metrics_base, metrics_ours])
    df.to_csv(output_dir / "comparison_metrics.csv", index=False)

    # Create summary panel from final trajectories.
    traj_base = np.load(output_dir / "baseline" / "trajectories.npy")
    traj_ours = np.load(output_dir / "ours" / "trajectories.npy")

    workspace = Workspace(
        xlim=tuple(cfg["workspace"]["xlim"]),
        ylim=tuple(cfg["workspace"]["ylim"]),
        resolution=int(cfg["workspace"]["resolution"]),
    )
    scenarios = default_scenarios()
    if cfg["scenario"]["name"] in scenarios:
        scn = scenarios[cfg["scenario"]["name"]]
    else:
        scn = {
            "hotspots": cfg["scenario"]["hotspots"],
            "obstacles": cfg["scenario"]["obstacles"],
        }
    X, Y, obstacle_mask = build_obstacle_mask(workspace, scn["obstacles"])
    q = gaussian_mixture_density(X, Y, scn["hotspots"])
    q = q / np.maximum(np.sum(q) * workspace.dx * workspace.dy, 1e-12)
    plot_comparison_panel(traj_base, traj_ours, q, obstacle_mask, workspace, output_dir / "baseline_vs_ours.png")
    plot_comparison_panel(traj_base, traj_ours, q, obstacle_mask, workspace, output_dir / "baseline_vs_ours_2agents.png")

    note = (
        "# Baseline vs Ours\n\n"
        f"- baseline overlap: {metrics_base['overlap_score']:.6f}\n"
        f"- ours overlap: {metrics_ours['overlap_score']:.6f}\n"
        f"- baseline ergodic: {metrics_base['ergodic_metric']:.6f}\n"
        f"- ours ergodic: {metrics_ours['ergodic_metric']:.6f}\n"
    )
    (output_dir / "experiment_note.md").write_text(note, encoding="utf-8")
    return df


def run_ablation(cfg: Dict, output_dir: Path) -> pd.DataFrame:
    """Run required ablation set."""
    ensure_dir(output_dir)
    runs: List[Dict[str, float]] = []

    cases = [
        ("no_coupling", {"coupling": False, "obstacle": True, "gamma": cfg["stein"]["gamma"]}),
        ("no_obstacle_mask", {"coupling": True, "obstacle": False, "gamma": cfg["stein"]["gamma"]}),
        ("gamma_low", {"coupling": True, "obstacle": True, "gamma": cfg["stein"]["gamma"] * 0.5}),
        ("gamma_high", {"coupling": True, "obstacle": True, "gamma": cfg["stein"]["gamma"] * 2.0}),
        ("alpha_low", {"coupling": True, "obstacle": True, "alpha": cfg["stein"]["alpha"] * 0.6}),
        ("alpha_large", {"coupling": True, "obstacle": True, "alpha": cfg["stein"]["alpha"] * 2.0}),
        ("kernel_narrow", {"coupling": True, "obstacle": True, "self_bandwidth": cfg["stein"]["self_bandwidth"] * 0.6}),
        ("kernel_wide", {"coupling": True, "obstacle": True, "self_bandwidth": cfg["stein"]["self_bandwidth"] * 2.0}),
        ("step_small", {"coupling": True, "obstacle": True, "flow_step_size": cfg["optimization"]["flow_step_size"] * 0.65}),
        ("step_large", {"coupling": True, "obstacle": True, "flow_step_size": cfg["optimization"]["flow_step_size"] * 1.35}),
        ("smooth_small", {"coupling": True, "obstacle": True, "traj_smooth_lambda": cfg["optimization"]["traj_smooth_lambda"] * 0.6}),
        ("smooth_large", {"coupling": True, "obstacle": True, "traj_smooth_lambda": cfg["optimization"]["traj_smooth_lambda"] * 1.4}),
    ]

    for name, opts in cases:
        c = copy.deepcopy(cfg)
        if "gamma" in opts:
            c["stein"]["gamma"] = float(opts["gamma"])
        if "alpha" in opts:
            c["stein"]["alpha"] = float(opts["alpha"])
        if "self_bandwidth" in opts:
            c["stein"]["self_bandwidth"] = float(opts["self_bandwidth"])
        if "flow_step_size" in opts:
            c["optimization"]["flow_step_size"] = float(opts["flow_step_size"])
        if "traj_smooth_lambda" in opts:
            c["optimization"]["traj_smooth_lambda"] = float(opts["traj_smooth_lambda"])

        m = run_single_variant(
            c,
            output_dir / name,
            name,
            use_coupling=bool(opts["coupling"]),
            use_obstacle_mask=bool(opts["obstacle"]),
        )
        m["ablation_case"] = name
        runs.append(m)

    df = pd.DataFrame(runs)
    df.to_csv(output_dir / "ablation_results.csv", index=False)

    plot_curves(
        {row["ablation_case"]: [row["ergodic_metric"]] for _, row in df.iterrows()},
        "final ergodic",
        "Ablation Ergodic Metric (final)",
        output_dir / "ablation_ergodic.png",
    )
    plot_curves(
        {row["ablation_case"]: [row["overlap_score"]] for _, row in df.iterrows()},
        "final overlap",
        "Ablation Overlap (final)",
        output_dir / "ablation_overlap.png",
    )
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        axes[0].bar(df["ablation_case"], df["ergodic_metric"])
        axes[0].set_title("Ablation Ergodic Metric")
        axes[0].tick_params(axis="x", rotation=60)
        axes[1].bar(df["ablation_case"], df["overlap_score"])
        axes[1].set_title("Ablation Overlap")
        axes[1].tick_params(axis="x", rotation=60)
        fig.savefig(output_dir / "ablation_plots.png")
        plt.close(fig)
    except Exception:
        pass

    (output_dir / "experiment_note.md").write_text(
        "# Ablation summary\n\nSee `ablation_results.csv` for full metrics and figures for each case.",
        encoding="utf-8",
    )
    return df


def run_compare_two_configs(
    baseline_cfg: Dict,
    ours_cfg: Dict,
    output_dir: Path,
    comparison_name: str = "baseline_vs_ours",
) -> pd.DataFrame:
    """Run baseline and ours from two explicit configs and export comparison."""
    ensure_dir(output_dir)
    metrics_base = run_single_variant(
        baseline_cfg,
        output_dir / "baseline",
        "baseline",
        use_coupling=False,
        use_obstacle_mask=bool(baseline_cfg.get("stein", {}).get("use_obstacle_mask", True)),
    )
    metrics_ours = run_single_variant(
        ours_cfg,
        output_dir / "ours",
        "ours",
        use_coupling=True,
        use_obstacle_mask=bool(ours_cfg.get("stein", {}).get("use_obstacle_mask", True)),
    )
    df = pd.DataFrame([metrics_base, metrics_ours])
    df.to_csv(output_dir / "comparison_metrics.csv", index=False)
    df.to_csv(output_dir / "metrics.csv", index=False)

    traj_base = np.load(output_dir / "baseline" / "trajectories.npy")
    traj_ours = np.load(output_dir / "ours" / "trajectories.npy")
    cfg_ref = ours_cfg
    workspace = Workspace(
        xlim=tuple(cfg_ref["workspace"]["xlim"]),
        ylim=tuple(cfg_ref["workspace"]["ylim"]),
        resolution=int(cfg_ref["workspace"]["resolution"]),
    )
    if cfg_ref["scenario"]["name"] == "mars_auto":
        mars_cfg = cfg_ref.get("mars", {})
        mars_data = build_mars_scene_from_dem(
            dem_path=mars_cfg["dem_path"],
            workspace=workspace,
            crop_size=int(mars_cfg.get("crop_size", 720)),
            slope_percentile=float(mars_cfg.get("slope_percentile", 88.0)),
            obstacle_dilation=int(mars_cfg.get("obstacle_dilation", 1)),
            complexity=str(mars_cfg.get("complexity", "medium")),
            crop_y0=mars_cfg.get("crop_y0", None),
            crop_x0=mars_cfg.get("crop_x0", None),
            obstacle_opening=int(mars_cfg.get("obstacle_opening", 0)),
            min_component_pixels=int(mars_cfg.get("min_component_pixels", 0)),
            density_smooth_sigma=float(mars_cfg.get("density_smooth_sigma", 2.0)),
            hotspot_count=int(mars_cfg.get("hotspot_count", 2)),
            hotspot_sigma=float(mars_cfg.get("hotspot_sigma", 0.08)),
            hotspot_blend=float(mars_cfg.get("hotspot_blend", 0.45)),
        )
        q = mars_data.base_density / np.maximum(np.sum(mars_data.base_density) * workspace.dx * workspace.dy, 1e-12)
        obstacle_mask = mars_data.obstacle_mask
    else:
        scn = default_scenarios()[cfg_ref["scenario"]["name"]]
        X, Y, obstacle_mask = build_obstacle_mask(workspace, scn["obstacles"])
        q = gaussian_mixture_density(X, Y, scn["hotspots"])
        q = q / np.maximum(np.sum(q) * workspace.dx * workspace.dy, 1e-12)
    plot_comparison_panel(traj_base, traj_ours, q, obstacle_mask, workspace, output_dir / f"{comparison_name}.png")
    save_json(
        output_dir / "metrics.json",
        {
            "baseline_ergodic_metric": float(metrics_base["ergodic_metric"]),
            "ours_ergodic_metric": float(metrics_ours["ergodic_metric"]),
            "baseline_overlap_score": float(metrics_base["overlap_score"]),
            "ours_overlap_score": float(metrics_ours["overlap_score"]),
        },
    )
    (output_dir / "experiment_note.md").write_text(
        f"# {comparison_name}\n\n"
        f"- baseline ergodic: {metrics_base['ergodic_metric']:.6f}\n"
        f"- ours ergodic: {metrics_ours['ergodic_metric']:.6f}\n"
        f"- baseline overlap: {metrics_base['overlap_score']:.6f}\n"
        f"- ours overlap: {metrics_ours['overlap_score']:.6f}\n",
        encoding="utf-8",
    )
    return df


def run_experiment_from_config(config_path: Path, output_dir: Path, mode: str) -> None:
    """Top-level helper used by scripts."""
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    ensure_dir(output_dir)

    if mode == "single":
        run_single_variant(cfg, output_dir, "single", use_coupling=cfg["stein"]["use_coupling"], use_obstacle_mask=cfg["stein"]["use_obstacle_mask"])
    elif mode == "compare":
        run_baseline_vs_ours(cfg, output_dir)
    elif mode == "ablation":
        run_ablation(cfg, output_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")

"""Optimization routines for flow-matching ergodic coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .density_utils import DensityField
from .ergodic_metric import summarize_metrics
from .map_utils import Workspace
from .rollout import rollout_multi_agent
from .sdf_utils import bilinear_interpolate_grid, query_sdf, sdf_gradient
from .stein_flow import FlowConfig, SteinTerms, multi_agent_stein_flow
from .trajectory_init import initialize_straight_line_trajectory
from .trajectory_update import update_trajectory_points


@dataclass
class OptimizationResult:
    """Outputs from optimization loop."""

    states: np.ndarray
    trajectories: np.ndarray
    controls: np.ndarray
    metrics_last: Dict[str, float]
    metric_curve: Dict[str, List[float]]
    snapshots: Dict[int, np.ndarray]
    stein_terms_last: List[SteinTerms]


def _infer_goals(cfg: Dict, initial_states: np.ndarray) -> np.ndarray:
    """Infer terminal goals from config for line-trajectory initialization."""
    n_agents = initial_states.shape[0]
    agents_cfg = cfg.get("agents", {})
    if "goal_positions" in agents_cfg:
        goals = np.asarray(agents_cfg["goal_positions"], dtype=float)
        if goals.shape == (n_agents, 2):
            return goals

    hotspots = cfg.get("scenario", {}).get("hotspots", [])
    if hotspots:
        weights = np.array([float(h.get("weight", 1.0)) for h in hotspots], dtype=float)
        means = np.array([h["mean"] for h in hotspots], dtype=float)
        wsum = float(np.sum(weights))
        if wsum <= 1e-12:
            weights = np.ones_like(weights)
            wsum = float(len(weights))
        centroid = np.sum(means * (weights[:, None] / wsum), axis=0)
        return np.repeat(centroid[None, :], n_agents, axis=0)

    center = np.array(
        [
            0.5 * (cfg["workspace"]["xlim"][0] + cfg["workspace"]["xlim"][1]),
            0.5 * (cfg["workspace"]["ylim"][0] + cfg["workspace"]["ylim"][1]),
        ],
        dtype=float,
    )
    return np.repeat(center[None, :], n_agents, axis=0)


def _line_trajectories(initial_states: np.ndarray, goals: np.ndarray, horizon: int) -> np.ndarray:
    """Build straight-line position trajectories from start to goal."""
    n_agents = initial_states.shape[0]
    traj = np.zeros((n_agents, horizon + 1, 2), dtype=float)
    for i in range(n_agents):
        traj[i] = initialize_straight_line_trajectory(initial_states[i, :2], goals[i], horizon)
    return traj


def _traj_to_states_controls(
    trajectories: np.ndarray,
    dt: float,
    max_accel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert position trajectories to state/control arrays."""
    n_agents, n_steps, _ = trajectories.shape
    horizon = n_steps - 1
    states = np.zeros((n_agents, n_steps, 4), dtype=float)
    controls = np.zeros((n_agents, horizon, 2), dtype=float)

    states[:, :, :2] = trajectories
    vel = np.diff(trajectories, axis=1) / dt
    states[:, 1:, 2:4] = vel
    states[:, 0, 2:4] = vel[:, 0, :]

    if horizon > 1:
        acc = np.diff(states[:, :, 2:4], axis=1) / dt
        controls[:, 1:, :] = acc[:, :-1, :]
        controls[:, 0, :] = acc[:, 0, :]
    norms = np.linalg.norm(controls, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_accel / np.maximum(norms, 1e-8))
    controls = controls * scale
    return states, controls


def _project_points_to_sdf_margin(
    points: np.ndarray,
    sdf: np.ndarray,
    workspace: Workspace,
    sdf_gx: np.ndarray,
    sdf_gy: np.ndarray,
    margin: float,
    n_steps: int = 2,
) -> np.ndarray:
    """Project points away from obstacles so sdf(points) >= margin approximately."""
    out = points.copy()
    for _ in range(max(int(n_steps), 1)):
        svals = query_sdf(out, sdf, workspace)
        viol = svals < margin
        if not np.any(viol):
            break
        gx = bilinear_interpolate_grid(out[:, 0], out[:, 1], sdf_gx, workspace)
        gy = bilinear_interpolate_grid(out[:, 0], out[:, 1], sdf_gy, workspace)
        g = np.stack([gx, gy], axis=-1)
        gnorm = np.linalg.norm(g, axis=-1, keepdims=True)
        gdir = g / np.maximum(gnorm, 1e-8)
        depth = (margin - svals)[:, None]
        out[viol] = out[viol] + (depth[viol] + 0.002) * gdir[viol]
        out[:, 0] = np.clip(out[:, 0], workspace.xlim[0], workspace.xlim[1])
        out[:, 1] = np.clip(out[:, 1], workspace.ylim[0], workspace.ylim[1])
    return out


def _smooth_controls(controls: np.ndarray, alpha: float) -> np.ndarray:
    """Temporal exponential smoothing of control sequence."""
    out = controls.copy()
    for t in range(1, out.shape[1]):
        out[:, t, :] = alpha * out[:, t - 1, :] + (1.0 - alpha) * out[:, t, :]
    return out


def _smooth_sequence(points: np.ndarray, passes: int) -> np.ndarray:
    """Light temporal smoothing for vector sequences."""
    out = points.copy()
    for _ in range(max(int(passes), 0)):
        cur = out.copy()
        cur[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = cur
    return out


def _repair_segments_against_obstacles(
    trajectory: np.ndarray,
    sdf: np.ndarray,
    workspace: Workspace,
    sdf_gx: np.ndarray,
    sdf_gy: np.ndarray,
    margin: float,
    passes: int = 2,
) -> np.ndarray:
    """Push segment endpoints when sampled segment points violate SDF margin."""
    out = trajectory.copy()
    if out.shape[0] < 3:
        return out

    sample_lambda = np.array([0.25, 0.5, 0.75], dtype=float)
    for _ in range(max(int(passes), 0)):
        changed = False
        for t in range(out.shape[0] - 1):
            a = out[t]
            b = out[t + 1]
            seg = (1.0 - sample_lambda[:, None]) * a[None, :] + sample_lambda[:, None] * b[None, :]
            svals = query_sdf(seg, sdf, workspace)
            viol = svals < margin
            if not np.any(viol):
                continue

            gx = bilinear_interpolate_grid(seg[:, 0], seg[:, 1], sdf_gx, workspace)
            gy = bilinear_interpolate_grid(seg[:, 0], seg[:, 1], sdf_gy, workspace)
            g = np.stack([gx, gy], axis=-1)
            gdir = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
            depth = (margin - svals)[:, None]
            push = np.sum((depth + 0.001) * gdir * viol[:, None], axis=0) / max(int(np.sum(viol)), 1)

            if t > 0:
                out[t] = out[t] + 0.45 * push
            if t + 1 < out.shape[0] - 1:
                out[t + 1] = out[t + 1] + 0.45 * push
            changed = True

        if not changed:
            break

        out[:, 0] = np.clip(out[:, 0], workspace.xlim[0], workspace.xlim[1])
        out[:, 1] = np.clip(out[:, 1], workspace.ylim[0], workspace.ylim[1])
    return out


def optimize_trajectories_approx(
    initial_states: np.ndarray,
    controls_init: np.ndarray,
    workspace: Workspace,
    density_field: DensityField,
    sdf: np.ndarray,
    cfg: Dict,
    use_coupling: bool,
    use_obstacle_mask: bool,
) -> OptimizationResult:
    """Engineering approximation optimizer.

    It tracks Stein reference flow via PD-style control updates.
    """
    dt = float(cfg["sim"]["dt"])
    max_accel = float(cfg["sim"]["max_accel"])
    iterations = int(cfg["optimization"]["iterations"])
    kp = float(cfg["optimization"]["kp"])
    kv = float(cfg["optimization"]["kv"])
    control_blend = float(cfg["optimization"]["control_blend"])
    smooth_alpha = float(cfg["optimization"]["smooth_alpha"])
    flow_step = float(cfg["optimization"]["flow_step_size"])
    snapshot_iters = sorted(set(int(v) for v in cfg["optimization"].get("snapshot_iters", [0, iterations])))
    obstacle_margin = float(cfg["optimization"].get("obstacle_margin", 0.05))
    obstacle_barrier_gain = float(cfg["optimization"].get("obstacle_barrier_gain", 1.2))
    inside_push_gain = float(cfg["optimization"].get("inside_push_gain", 12.0))
    inside_vel_damp = float(cfg["optimization"].get("inside_vel_damp", 1.6))

    flow_cfg = FlowConfig(
        self_bandwidth=float(cfg["stein"]["self_bandwidth"]),
        coupling_bandwidth=float(cfg["stein"]["coupling_bandwidth"]),
        gamma=float(cfg["stein"]["gamma"] if use_coupling else 0.0),
        use_coupling=use_coupling,
        task_weight=float(cfg["stein"].get("task_weight", 1.0)),
        self_repulsion_weight=float(cfg["stein"].get("self_repulsion_weight", 1.0)),
        strict_paper=bool(cfg["stein"].get("strict_paper", False)),
    )

    controls = controls_init.copy()
    bounds = (tuple(cfg["workspace"]["xlim"]), tuple(cfg["workspace"]["ylim"]))

    curve: Dict[str, List[float]] = {
        "ergodic_metric": [],
        "overlap_score": [],
        "obstacle_violation_ratio": [],
    }
    snapshots: Dict[int, np.ndarray] = {}

    states = trajectories = None
    stein_terms: List[SteinTerms] = []
    metrics: Dict[str, float] = {}
    sdf_gx, sdf_gy = sdf_gradient(sdf, workspace)

    for it in tqdm(range(iterations + 1), desc="opt", leave=False):
        states, trajectories = rollout_multi_agent(initial_states, controls, dt, max_accel, bounds)
        stein_terms = multi_agent_stein_flow(trajectories, density_field, workspace, flow_cfg)

        metrics, _ = summarize_metrics(
            trajectories=trajectories,
            controls=controls,
            q_grid=density_field.q,
            workspace=workspace,
            sdf=sdf,
            dt=dt,
            fourier_K=int(cfg["metrics"]["fourier_K"]),
            overlap_sigma=float(cfg["metrics"]["overlap_sigma"]),
        )

        for k in curve:
            curve[k].append(float(metrics[k]))
        if it in snapshot_iters:
            snapshots[it] = trajectories.copy()

        if it == iterations:
            break

        new_controls = controls.copy()
        for i in range(trajectories.shape[0]):
            p = trajectories[i, :-1, :]
            v = states[i, :-1, 2:4]
            href = stein_terms[i].total[:-1, :]
            if use_obstacle_mask:
                # Barrier correction from SDF gradient to discourage penetration.
                svals = query_sdf(p, sdf, workspace)
                gx = bilinear_interpolate_grid(p[:, 0], p[:, 1], sdf_gx, workspace)
                gy = bilinear_interpolate_grid(p[:, 0], p[:, 1], sdf_gy, workspace)
                grad = np.stack([gx, gy], axis=-1)
                grad_dir = grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-8)
                strength = np.clip((obstacle_margin - svals) / max(obstacle_margin, 1e-8), 0.0, 2.0)
                href = href + obstacle_barrier_gain * strength[:, None] * grad_dir
            p_des = p + flow_step * href
            v_des = href
            a_ref = kp * (p_des - p) + kv * (v_des - v)
            if use_obstacle_mask:
                # Hard corrective push if trajectory points are already inside obstacles.
                inside = svals < 0.0
                if np.any(inside):
                    depth = (-svals[inside])[:, None]
                    a_ref[inside] += inside_push_gain * (depth + 0.02) * grad_dir[inside]
                    a_ref[inside] -= inside_vel_damp * v[inside]
            new_controls[i, :, :] = (1.0 - control_blend) * controls[i, :, :] + control_blend * a_ref

        controls = _smooth_controls(new_controls, smooth_alpha)

    assert states is not None and trajectories is not None
    return OptimizationResult(
        states=states,
        trajectories=trajectories,
        controls=controls,
        metrics_last=metrics,
        metric_curve=curve,
        snapshots=snapshots,
        stein_terms_last=stein_terms,
    )


def optimize_trajectories_lqfm_fallback(*args, **kwargs) -> OptimizationResult:
    """Placeholder for LQFM-style mode.

    In this version we fallback to approximate optimizer while preserving mode separation.
    """
    return optimize_trajectories_approx(*args, **kwargs)


def optimize_trajectories_line_svgd(
    initial_states: np.ndarray,
    controls_init: np.ndarray,
    workspace: Workspace,
    density_field: DensityField,
    sdf: np.ndarray,
    cfg: Dict,
    use_coupling: bool,
    use_obstacle_mask: bool,
) -> OptimizationResult:
    """Stein iteration directly on line-initialized trajectory particles.

    x_{t}^{k+1} = x_{t}^{k} + eta * phi(x_{t}^{k}), initialized from straight line.
    """
    del controls_init  # Unused in this mode by design.
    dt = float(cfg["sim"]["dt"])
    max_accel = float(cfg["sim"]["max_accel"])
    iterations = int(cfg["optimization"]["iterations"])
    eta = float(cfg["optimization"].get("flow_step_size", 0.05))
    traj_smooth = float(cfg["optimization"].get("traj_smooth_lambda", 0.15))
    line_anchor_lambda = float(cfg["optimization"].get("line_anchor_lambda", 0.18))
    line_anchor_decay = float(cfg["optimization"].get("line_anchor_decay", 0.996))
    line_tube_radius = float(cfg["optimization"].get("line_tube_radius", 0.22))
    max_particle_step = float(cfg["optimization"].get("max_particle_step", 0.012))
    smooth_passes = int(cfg["optimization"].get("trajectory_smooth_passes", 2))
    smooth_blend = float(cfg["optimization"].get("trajectory_smooth_blend", 0.5))
    enforce_monotonic = bool(cfg["optimization"].get("enforce_monotonic_progress", True))
    fix_endpoints = bool(cfg["optimization"].get("fix_line_endpoints", False))
    post_smooth_iters = int(cfg["optimization"].get("post_smooth_iters", 0))
    post_smooth_lambda = float(cfg["optimization"].get("post_smooth_lambda", 0.25))
    max_segment_length = float(cfg["optimization"].get("max_segment_length", 0.0))
    segment_repair_passes = int(cfg["optimization"].get("segment_repair_passes", 2))
    obstacle_margin = float(cfg["optimization"].get("obstacle_margin", 0.08))
    obstacle_barrier_gain = float(cfg["optimization"].get("obstacle_barrier_gain", 3.0))
    projection_steps = int(cfg["optimization"].get("projection_steps", 5))
    projection_margin_ratio = float(cfg["optimization"].get("projection_margin_ratio", 0.8))
    flow_smooth_passes = int(cfg["optimization"].get("flow_smooth_passes", 1))
    snapshot_iters = sorted(set(int(v) for v in cfg["optimization"].get("snapshot_iters", [0, iterations])))
    horizon = int(cfg["sim"]["horizon"])
    strict_paper = bool(cfg.get("stein", {}).get("strict_paper", False))

    if strict_paper:
        # Strict paper-mode: keep trajectory-based Stein update only.
        line_anchor_lambda = 0.0
        line_anchor_decay = 1.0
        line_tube_radius = 1e6
        max_particle_step = 1e9
        smooth_passes = 0
        smooth_blend = 0.0
        enforce_monotonic = False
        post_smooth_iters = 0
        max_segment_length = 0.0
        segment_repair_passes = 0
        flow_smooth_passes = 0

    goals = _infer_goals(cfg, initial_states)
    trajectories = _line_trajectories(initial_states, goals, horizon)
    line_ref = trajectories.copy()
    bounds = (tuple(cfg["workspace"]["xlim"]), tuple(cfg["workspace"]["ylim"]))
    sdf_gx, sdf_gy = sdf_gradient(sdf, workspace)

    flow_cfg = FlowConfig(
        self_bandwidth=float(cfg["stein"]["self_bandwidth"]),
        coupling_bandwidth=float(cfg["stein"]["coupling_bandwidth"]),
        gamma=float(cfg["stein"]["gamma"] if use_coupling else 0.0),
        use_coupling=use_coupling,
        task_weight=float(cfg["stein"].get("task_weight", 1.0)),
        self_repulsion_weight=float(cfg["stein"].get("self_repulsion_weight", 1.0)),
        strict_paper=bool(cfg["stein"].get("strict_paper", False)),
    )

    curve: Dict[str, List[float]] = {
        "ergodic_metric": [],
        "overlap_score": [],
        "obstacle_violation_ratio": [],
    }
    snapshots: Dict[int, np.ndarray] = {}
    stein_terms: List[SteinTerms] = []
    metrics: Dict[str, float] = {}

    for it in tqdm(range(iterations + 1), desc="line-svgd", leave=False):
        stein_terms = multi_agent_stein_flow(trajectories, density_field, workspace, flow_cfg)
        states, controls = _traj_to_states_controls(trajectories, dt=dt, max_accel=max_accel)
        metrics, _ = summarize_metrics(
            trajectories=trajectories,
            controls=controls,
            q_grid=density_field.q,
            workspace=workspace,
            sdf=sdf,
            dt=dt,
            fourier_K=int(cfg["metrics"]["fourier_K"]),
            overlap_sigma=float(cfg["metrics"]["overlap_sigma"]),
        )
        for k in curve:
            curve[k].append(float(metrics[k]))
        if it in snapshot_iters:
            snapshots[it] = trajectories.copy()
        if it == iterations:
            break

        for i in range(trajectories.shape[0]):
            flow = _smooth_sequence(stein_terms[i].total, flow_smooth_passes)
            flow[0] = 0.0
            if fix_endpoints:
                flow[-1] = 0.0
            dnorm = np.linalg.norm(flow, axis=-1, keepdims=True)
            flow = flow * np.minimum(1.0, max_particle_step / np.maximum(eta * dnorm, 1e-8))
            trajectories[i] = update_trajectory_points(
                trajectories[i],
                flow,
                step_size=eta,
                smooth_lambda=traj_smooth,
                fix_start=True,
                fix_end=fix_endpoints,
            )
            if smooth_passes > 0:
                smoothed = trajectories[i].copy()
                for _ in range(smooth_passes):
                    cur = smoothed.copy()
                    cur[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
                    smoothed = cur
                trajectories[i, 1:-1] = (
                    (1.0 - smooth_blend) * trajectories[i, 1:-1] + smooth_blend * smoothed[1:-1]
                )

            # Keep the iteration close to a straight-line initialization tube.
            anchor_w = line_anchor_lambda * (line_anchor_decay ** it)
            trajectories[i, 1:-1] = (1.0 - anchor_w) * trajectories[i, 1:-1] + anchor_w * line_ref[i, 1:-1]
            off = trajectories[i, 1:-1] - line_ref[i, 1:-1]
            off_norm = np.linalg.norm(off, axis=-1, keepdims=True)
            off_scale = np.minimum(1.0, line_tube_radius / np.maximum(off_norm, 1e-8))
            trajectories[i, 1:-1] = line_ref[i, 1:-1] + off * off_scale

            if enforce_monotonic:
                p0 = initial_states[i, :2]
                d = goals[i] - p0
                d2 = float(np.dot(d, d))
                if d2 > 1e-10:
                    raw = trajectories[i].copy()
                    s_raw = np.sum((raw - p0[None, :]) * d[None, :], axis=1) / d2
                    s_int = np.clip(np.maximum.accumulate(s_raw[1:-1]), 0.0, 1.0)
                    base_new = p0[None, :] + s_int[:, None] * d[None, :]
                    base_old = p0[None, :] + s_raw[1:-1][:, None] * d[None, :]
                    lateral = raw[1:-1] - base_old
                    trajectories[i, 1:-1] = base_new + lateral

            if use_obstacle_mask:
                p = trajectories[i, 1:-1]
                svals = query_sdf(p, sdf, workspace)
                gx = bilinear_interpolate_grid(p[:, 0], p[:, 1], sdf_gx, workspace)
                gy = bilinear_interpolate_grid(p[:, 0], p[:, 1], sdf_gy, workspace)
                g = np.stack([gx, gy], axis=-1)
                gdir = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
                strength = np.clip((obstacle_margin - svals) / max(obstacle_margin, 1e-8), 0.0, 3.0)
                trajectories[i, 1:-1] = p + eta * obstacle_barrier_gain * strength[:, None] * gdir
                trajectories[i, 1:-1] = _project_points_to_sdf_margin(
                    trajectories[i, 1:-1],
                    sdf=sdf,
                    workspace=workspace,
                    sdf_gx=sdf_gx,
                    sdf_gy=sdf_gy,
                    margin=max(0.012, projection_margin_ratio * obstacle_margin),
                    n_steps=projection_steps,
                )

            trajectories[i, :, 0] = np.clip(trajectories[i, :, 0], bounds[0][0], bounds[0][1])
            trajectories[i, :, 1] = np.clip(trajectories[i, :, 1], bounds[1][0], bounds[1][1])
            trajectories[i, 0, :] = initial_states[i, :2]
            if fix_endpoints:
                trajectories[i, -1, :] = goals[i]
            if max_segment_length > 1e-8:
                for t in range(1, trajectories.shape[1]):
                    d = trajectories[i, t] - trajectories[i, t - 1]
                    n = float(np.linalg.norm(d))
                    if n > max_segment_length:
                        trajectories[i, t] = trajectories[i, t - 1] + d * (max_segment_length / n)
            if use_obstacle_mask:
                trajectories[i] = _repair_segments_against_obstacles(
                    trajectories[i],
                    sdf=sdf,
                    workspace=workspace,
                    sdf_gx=sdf_gx,
                    sdf_gy=sdf_gy,
                    margin=max(0.01, 0.75 * obstacle_margin),
                    passes=segment_repair_passes,
                )
                trajectories[i, 1:-1] = _project_points_to_sdf_margin(
                    trajectories[i, 1:-1],
                    sdf=sdf,
                    workspace=workspace,
                    sdf_gx=sdf_gx,
                    sdf_gy=sdf_gy,
                    margin=max(0.012, projection_margin_ratio * obstacle_margin),
                    n_steps=projection_steps,
                )
                trajectories[i, 0, :] = initial_states[i, :2]
                if fix_endpoints:
                    trajectories[i, -1, :] = goals[i]

    states, controls = _traj_to_states_controls(trajectories, dt=dt, max_accel=max_accel)
    if post_smooth_iters > 0:
        for _ in range(post_smooth_iters):
            prev = trajectories.copy()
            trajectories[:, 1:-1, :] = (
                (1.0 - post_smooth_lambda) * prev[:, 1:-1, :]
                + 0.5 * post_smooth_lambda * (prev[:, :-2, :] + prev[:, 2:, :])
            )
            for i in range(trajectories.shape[0]):
                if use_obstacle_mask:
                    trajectories[i, 1:-1] = _project_points_to_sdf_margin(
                        trajectories[i, 1:-1],
                        sdf=sdf,
                        workspace=workspace,
                        sdf_gx=sdf_gx,
                        sdf_gy=sdf_gy,
                        margin=max(0.012, projection_margin_ratio * obstacle_margin),
                        n_steps=max(2, projection_steps - 1),
                    )
                    trajectories[i] = _repair_segments_against_obstacles(
                        trajectories[i],
                        sdf=sdf,
                        workspace=workspace,
                        sdf_gx=sdf_gx,
                        sdf_gy=sdf_gy,
                        margin=max(0.01, 0.75 * obstacle_margin),
                        passes=segment_repair_passes,
                    )
                trajectories[i, :, 0] = np.clip(trajectories[i, :, 0], bounds[0][0], bounds[0][1])
                trajectories[i, :, 1] = np.clip(trajectories[i, :, 1], bounds[1][0], bounds[1][1])
                trajectories[i, 0, :] = initial_states[i, :2]
                if fix_endpoints:
                    trajectories[i, -1, :] = goals[i]
                if max_segment_length > 1e-8:
                    for t in range(1, trajectories.shape[1]):
                        d = trajectories[i, t] - trajectories[i, t - 1]
                        n = float(np.linalg.norm(d))
                        if n > max_segment_length:
                            trajectories[i, t] = trajectories[i, t - 1] + d * (max_segment_length / n)
        states, controls = _traj_to_states_controls(trajectories, dt=dt, max_accel=max_accel)
    return OptimizationResult(
        states=states,
        trajectories=trajectories,
        controls=controls,
        metrics_last=metrics,
        metric_curve=curve,
        snapshots=snapshots,
        stein_terms_last=stein_terms,
    )

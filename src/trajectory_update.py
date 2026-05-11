"""Trajectory update rules for Stein trajectory evolution."""

from __future__ import annotations

import numpy as np


def laplacian_smooth_term(trajectory: np.ndarray) -> np.ndarray:
    """Compute discrete second-order smoothness term for trajectory points."""
    out = np.zeros_like(trajectory)
    out[1:-1] = trajectory[:-2] - 2.0 * trajectory[1:-1] + trajectory[2:]
    return out


def update_trajectory_points(
    trajectory: np.ndarray,
    stein_direction: np.ndarray,
    step_size: float,
    smooth_lambda: float,
    fix_start: bool = True,
    fix_end: bool = False,
) -> np.ndarray:
    """Apply the required trajectory-point update.

    p_new(t) = p(t) + step_size * stein_dir(t) + smooth_lambda * Laplacian(p)(t)
    for t in 1..T-1, with optional fixed endpoints.
    """
    traj = np.asarray(trajectory, dtype=float)
    direc = np.asarray(stein_direction, dtype=float)
    if traj.shape != direc.shape:
        raise ValueError("trajectory and stein_direction must share the same shape")

    out = traj.copy()
    smooth = laplacian_smooth_term(traj)
    out[1:-1] = traj[1:-1] + float(step_size) * direc[1:-1] + float(smooth_lambda) * smooth[1:-1]
    if fix_start:
        out[0] = traj[0]
    if fix_end:
        out[-1] = traj[-1]
    return out


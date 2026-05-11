"""Fourier-based ergodic metrics and supporting metrics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .map_utils import Workspace
from .sdf_utils import query_sdf


@dataclass
class ErgodicResult:
    """Ergodic metric details."""

    metric: float
    c_k: np.ndarray
    phi_k: np.ndarray
    lambda_k: np.ndarray
    ks: np.ndarray


def _basis_values(points: np.ndarray, ks: np.ndarray, workspace: Workspace) -> np.ndarray:
    """Compute cosine basis values for points and wave vectors."""
    x = (points[:, 0] - workspace.xlim[0]) / (workspace.xlim[1] - workspace.xlim[0])
    y = (points[:, 1] - workspace.ylim[0]) / (workspace.ylim[1] - workspace.ylim[0])
    vals = np.cos(np.pi * x[:, None] * ks[None, :, 0]) * np.cos(np.pi * y[:, None] * ks[None, :, 1])
    return vals


def compute_ergodic_metric(
    trajectories: np.ndarray,
    q_grid: np.ndarray,
    workspace: Workspace,
    K: int = 8,
    s: float = 1.5,
) -> ErgodicResult:
    """Compute truncated Fourier ergodic metric for multi-agent trajectories."""
    ks = np.array(list(itertools.product(range(K + 1), range(K + 1))), dtype=float)

    pts = trajectories.reshape(-1, 2)
    basis_pts = _basis_values(pts, ks, workspace)
    c_k = np.mean(basis_pts, axis=0)

    X, Y = workspace.meshgrid()
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    basis_grid = _basis_values(grid_pts, ks, workspace)
    area = workspace.dx * workspace.dy
    phi_k = np.sum(basis_grid * q_grid.ravel()[:, None], axis=0) * area

    lambda_k = 1.0 / (1.0 + np.sum(ks**2, axis=1)) ** s
    metric = float(np.sum(lambda_k * (c_k - phi_k) ** 2))

    return ErgodicResult(metric=metric, c_k=c_k, phi_k=phi_k, lambda_k=lambda_k, ks=ks)


def path_length(trajectories: np.ndarray) -> float:
    """Total path length over all agents."""
    diff = np.diff(trajectories, axis=1)
    return float(np.sum(np.linalg.norm(diff, axis=-1)))


def control_energy(controls: np.ndarray, dt: float) -> float:
    """Integral-like control energy."""
    return float(np.sum(np.sum(controls**2, axis=-1)) * dt)


def smoothness(controls: np.ndarray) -> float:
    """Second-order smoothness proxy based on acceleration changes."""
    d = np.diff(controls, axis=1)
    return float(np.mean(np.linalg.norm(d, axis=-1) ** 2))


def obstacle_metrics(trajectories: np.ndarray, sdf: np.ndarray, workspace: Workspace) -> Dict[str, float]:
    """Obstacle violation and clearance metrics."""
    vals = query_sdf(trajectories, sdf, workspace)
    min_dist = float(np.min(vals))
    violation_ratio = float(np.mean(vals < 0.0))
    feasible_ratio = 1.0 - violation_ratio
    return {
        "min_obstacle_distance": min_dist,
        "obstacle_violation_ratio": violation_ratio,
        "feasible_ratio": feasible_ratio,
    }


def overlap_score(trajectories: np.ndarray, sigma: float = 0.05) -> float:
    """Average pairwise trajectory proximity over time."""
    n = trajectories.shape[0]
    if n < 2:
        return 0.0
    scores: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = np.sum((trajectories[i] - trajectories[j]) ** 2, axis=-1)
            scores.append(float(np.mean(np.exp(-d2 / (2.0 * sigma * sigma)))))
    return float(np.mean(scores))


def summarize_metrics(
    trajectories: np.ndarray,
    controls: np.ndarray,
    q_grid: np.ndarray,
    workspace: Workspace,
    sdf: np.ndarray,
    dt: float,
    fourier_K: int,
    overlap_sigma: float,
) -> Tuple[Dict[str, float], ErgodicResult]:
    """Compute all scalar metrics used in experiments."""
    ergo = compute_ergodic_metric(trajectories, q_grid, workspace, K=fourier_K)
    obs = obstacle_metrics(trajectories, sdf, workspace)

    metrics = {
        "ergodic_metric": ergo.metric,
        "path_length": path_length(trajectories),
        "control_energy": control_energy(controls, dt),
        "smoothness": smoothness(controls),
        "overlap_score": overlap_score(trajectories, sigma=overlap_sigma),
        **obs,
    }
    return metrics, ergo

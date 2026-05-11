"""Target density and obstacle-aware density utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import multivariate_normal

from .map_utils import Workspace
from .sdf_utils import bilinear_interpolate_grid


@dataclass
class DensityField:
    """Container for density grid and log-gradient grid."""

    q: np.ndarray
    logq: np.ndarray
    grad_logq_x: np.ndarray
    grad_logq_y: np.ndarray


def gaussian_mixture_density(
    X: np.ndarray,
    Y: np.ndarray,
    components: List[Dict],
) -> np.ndarray:
    """Evaluate 2D Gaussian mixture on meshgrid."""
    pos = np.stack([X, Y], axis=-1)
    q = np.zeros(X.shape, dtype=float)
    total_w = sum(c["weight"] for c in components)
    for comp in components:
        rv = multivariate_normal(mean=comp["mean"], cov=np.array(comp["cov"], dtype=float))
        q += (comp["weight"] / total_w) * rv.pdf(pos)
    return q


def normalize_density(q: np.ndarray, workspace: Workspace) -> np.ndarray:
    """Normalize discrete density over workspace."""
    area = workspace.dx * workspace.dy
    z = np.sum(q) * area
    if z <= 1e-12:
        raise ValueError("Density integral is too small.")
    return q / z


def build_density_field(q: np.ndarray, workspace: Workspace, eps: float = 1e-12) -> DensityField:
    """Build density and grad-log-density grids."""
    qn = normalize_density(np.maximum(q, eps), workspace)
    logq = np.log(np.maximum(qn, eps))
    grad_y, grad_x = np.gradient(logq, workspace.dy, workspace.dx)
    return DensityField(q=qn, logq=logq, grad_logq_x=grad_x, grad_logq_y=grad_y)


def obstacle_aware_density(
    base_q: np.ndarray,
    traversability_mask: np.ndarray,
    workspace: Workspace,
    eps: float = 1e-12,
) -> DensityField:
    """Compute q_tilde ~ q * m and its grad-log grid."""
    q_task = np.maximum(base_q * traversability_mask, eps)
    return build_density_field(q_task, workspace, eps=eps)


def query_density_field(points: np.ndarray, field: DensityField, workspace: Workspace) -> Tuple[np.ndarray, np.ndarray]:
    """Query q and grad log q for points (..., 2)."""
    pts = np.asarray(points)
    shp = pts.shape
    flat = pts.reshape(-1, 2)

    qv = bilinear_interpolate_grid(flat[:, 0], flat[:, 1], field.q, workspace)
    gx = bilinear_interpolate_grid(flat[:, 0], flat[:, 1], field.grad_logq_x, workspace)
    gy = bilinear_interpolate_grid(flat[:, 0], flat[:, 1], field.grad_logq_y, workspace)

    grad = np.stack([gx, gy], axis=-1).reshape(*shp[:-1], 2)
    return qv.reshape(shp[:-1]), grad

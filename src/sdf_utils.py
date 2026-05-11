"""Signed distance field utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from .map_utils import Workspace


def compute_sdf(obstacle_mask: np.ndarray, workspace: Workspace) -> np.ndarray:
    """Compute signed distance field from binary obstacle mask.

    free-space distance is positive; inside-obstacle distance is negative.
    """
    dx, dy = workspace.dx, workspace.dy
    spacing = (dy, dx)

    obstacle = obstacle_mask.astype(bool)
    free_dist = distance_transform_edt(~obstacle, sampling=spacing)
    inside_dist = distance_transform_edt(obstacle, sampling=spacing)
    sdf = free_dist - inside_dist
    return sdf


def bilinear_interpolate_grid(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    workspace: Workspace,
) -> np.ndarray:
    """Bilinear interpolation for regular grid values."""
    x = np.clip(x, workspace.xlim[0], workspace.xlim[1])
    y = np.clip(y, workspace.ylim[0], workspace.ylim[1])

    tx = (x - workspace.xlim[0]) / (workspace.xlim[1] - workspace.xlim[0]) * (workspace.resolution - 1)
    ty = (y - workspace.ylim[0]) / (workspace.ylim[1] - workspace.ylim[0]) * (workspace.resolution - 1)

    x0 = np.floor(tx).astype(int)
    y0 = np.floor(ty).astype(int)
    x1 = np.clip(x0 + 1, 0, workspace.resolution - 1)
    y1 = np.clip(y0 + 1, 0, workspace.resolution - 1)

    wx = tx - x0
    wy = ty - y0

    g00 = grid[y0, x0]
    g01 = grid[y1, x0]
    g10 = grid[y0, x1]
    g11 = grid[y1, x1]

    return (
        (1 - wx) * (1 - wy) * g00
        + (1 - wx) * wy * g01
        + wx * (1 - wy) * g10
        + wx * wy * g11
    )


def query_sdf(points: np.ndarray, sdf: np.ndarray, workspace: Workspace) -> np.ndarray:
    """Query SDF for points of shape (..., 2)."""
    pts = np.asarray(points)
    s = pts.shape
    flat = pts.reshape(-1, 2)
    vals = bilinear_interpolate_grid(flat[:, 0], flat[:, 1], sdf, workspace)
    return vals.reshape(s[:-1])


def traversability_mask_from_sdf(sdf: np.ndarray, alpha: float) -> np.ndarray:
    """Compute m(x)=sigmoid(sdf/alpha)."""
    z = np.clip(sdf / alpha, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def sdf_gradient(sdf: np.ndarray, workspace: Workspace) -> Tuple[np.ndarray, np.ndarray]:
    """Compute numerical gradient of SDF on grid."""
    gy, gx = np.gradient(sdf, workspace.dy, workspace.dx)
    return gx, gy

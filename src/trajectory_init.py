"""Trajectory initialization utilities (trajectory-first, not random particles)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def initialize_straight_line_trajectory(
    start: Sequence[float],
    goal: Sequence[float],
    T: int,
) -> np.ndarray:
    """Initialize one trajectory as a straight line from start to goal.

    Returns:
        ndarray with shape [T+1, 2]
    """
    p0 = np.asarray(start, dtype=float)
    pg = np.asarray(goal, dtype=float)
    s = np.linspace(0.0, 1.0, int(T) + 1)[:, None]
    return (1.0 - s) * p0[None, :] + s * pg[None, :]


def initialize_constant_velocity_rollout(
    start: Sequence[float],
    velocity: Sequence[float],
    T: int,
    dt: float,
) -> np.ndarray:
    """Initialize one trajectory by constant velocity rollout.

    Returns:
        ndarray with shape [T+1, 2]
    """
    p0 = np.asarray(start, dtype=float)
    v = np.asarray(velocity, dtype=float)
    t = np.arange(int(T) + 1, dtype=float)[:, None]
    return p0[None, :] + t * float(dt) * v[None, :]


def initialize_polyline_trajectory(
    waypoints: Iterable[Sequence[float]],
    T: int,
) -> np.ndarray:
    """Initialize one trajectory by piecewise-linear interpolation on waypoints.

    Args:
        waypoints: at least two 2D points.
        T: number of steps (trajectory has T+1 points).
    """
    wp = np.asarray(list(waypoints), dtype=float)
    if wp.shape[0] < 2:
        raise ValueError("waypoints must contain at least two points")

    seg = np.diff(wp, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total <= 1e-12:
        return np.repeat(wp[:1], int(T) + 1, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg_len / total)])
    u = np.linspace(0.0, 1.0, int(T) + 1)
    out = np.zeros((int(T) + 1, 2), dtype=float)
    for i, ui in enumerate(u):
        k = int(np.clip(np.searchsorted(cum, ui, side="right") - 1, 0, len(seg_len) - 1))
        l = cum[k + 1] - cum[k]
        a = 0.0 if l <= 1e-12 else float((ui - cum[k]) / l)
        out[i] = (1.0 - a) * wp[k] + a * wp[k + 1]
    return out


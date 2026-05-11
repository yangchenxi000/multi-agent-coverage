"""Double-integrator dynamics utilities."""

from __future__ import annotations

import numpy as np


def step_double_integrator(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """One-step update for state [px, py, vx, vy]."""
    px, py, vx, vy = state
    ax, ay = control
    next_state = np.array(
        [
            px + vx * dt,
            py + vy * dt,
            vx + ax * dt,
            vy + ay * dt,
        ],
        dtype=float,
    )
    return next_state


def clip_control(controls: np.ndarray, max_accel: float) -> np.ndarray:
    """Clip control norm by max acceleration."""
    norms = np.linalg.norm(controls, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_accel / np.maximum(norms, 1e-8))
    return controls * scale

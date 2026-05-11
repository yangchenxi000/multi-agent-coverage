"""Rollout trajectories under double-integrator dynamics."""

from __future__ import annotations

import numpy as np

from .dynamics import clip_control, step_double_integrator


def rollout_multi_agent(
    initial_states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    max_accel: float,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out states and positions for all agents.

    initial_states: (N, 4)
    controls: (N, T, 2)
    returns:
      states: (N, T+1, 4)
      positions: (N, T+1, 2)
    """
    n_agents, T, _ = controls.shape
    states = np.zeros((n_agents, T + 1, 4), dtype=float)
    states[:, 0, :] = initial_states
    controls_clipped = clip_control(controls, max_accel)

    for t in range(T):
        for i in range(n_agents):
            states[i, t + 1, :] = step_double_integrator(states[i, t, :], controls_clipped[i, t, :], dt)
            states[i, t + 1, 0] = np.clip(states[i, t + 1, 0], workspace_bounds[0][0], workspace_bounds[0][1])
            states[i, t + 1, 1] = np.clip(states[i, t + 1, 1], workspace_bounds[1][0], workspace_bounds[1][1])

    return states, states[:, :, :2]

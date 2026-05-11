"""Stein reference flow for single and multi-agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .density_utils import DensityField, query_density_field
from .kernels import grad_x_rbf_kernel, rbf_kernel
from .map_utils import Workspace


@dataclass
class SteinTerms:
    """Decomposed Stein terms per trajectory point."""

    task_term: np.ndarray
    self_repulsion_term: np.ndarray
    coupling_term: np.ndarray
    total: np.ndarray


@dataclass
class FlowConfig:
    """Stein flow hyper-parameters."""

    self_bandwidth: float
    coupling_bandwidth: float
    gamma: float
    use_coupling: bool
    task_weight: float = 1.0
    self_repulsion_weight: float = 1.0
    strict_paper: bool = False


def _self_stein_term(
    points: np.ndarray,
    density_field: DensityField,
    workspace: Workspace,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute task and self-repulsion terms for one trajectory."""
    _, grad_log_q = query_density_field(points, density_field, workspace)
    if bandwidth <= 0.0:
        raise ValueError("self_bandwidth must be positive in strict Stein flow")
    K = rbf_kernel(points, points, bandwidth)

    # Task attraction term: E_j[k(x_j, x_i) grad log q(x_j)]
    task = (K.T @ grad_log_q) / points.shape[0]

    # Self repulsion term: E_j[grad_{x_j} k(x_j, x_i)]
    grad_tensor = grad_x_rbf_kernel(points, points, bandwidth)
    self_rep = np.mean(np.transpose(grad_tensor, (1, 0, 2)), axis=1)
    return task, self_rep


def _coupling_repulsion(
    points_i: np.ndarray,
    points_other: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Cross-trajectory repulsive coupling using kernel gradients."""
    if points_other.size == 0:
        return np.zeros_like(points_i)
    K = rbf_kernel(points_other, points_i, bandwidth)
    diff = points_i[None, :, :] - points_other[:, None, :]
    rep = (2.0 / max(bandwidth, 1e-8)) * np.mean(K[:, :, None] * diff, axis=0)
    return rep


def multi_agent_stein_flow(
    trajectories: np.ndarray,
    density_field: DensityField,
    workspace: Workspace,
    flow_cfg: FlowConfig,
) -> List[SteinTerms]:
    """Compute decomposed Stein flow for each agent trajectory.

    trajectories shape: (n_agents, T+1, 2)
    """
    n_agents = trajectories.shape[0]
    terms: List[SteinTerms] = []

    for i in range(n_agents):
        Xi = trajectories[i]
        task_term, self_term = _self_stein_term(
            Xi,
            density_field,
            workspace,
            flow_cfg.self_bandwidth,
        )

        if flow_cfg.use_coupling and n_agents > 1:
            if flow_cfg.coupling_bandwidth <= 0.0:
                raise ValueError("coupling_bandwidth must be positive in strict Stein flow")
            # Eq.(22): gamma * sum_{l!=i} E_{x'~mu_{P_l}}[grad_{x'} k'(x', x)]
            # Keep per-agent expectation then sum, instead of averaging over all others.
            coupling = np.zeros_like(Xi)
            for l in range(n_agents):
                if l == i:
                    continue
                coupling += _coupling_repulsion(Xi, trajectories[l], flow_cfg.coupling_bandwidth)
        else:
            coupling = np.zeros_like(Xi)

        task = flow_cfg.task_weight * task_term
        self_rep = flow_cfg.self_repulsion_weight * self_term
        cpl = flow_cfg.gamma * coupling
        total = (
            task
            + self_rep
            + cpl
        )
        terms.append(
            SteinTerms(
                task_term=task,
                self_repulsion_term=self_rep,
                coupling_term=cpl,
                total=total,
            )
        )
    return terms

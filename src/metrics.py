"""Metrics API wrapper for experiment code."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .ergodic_metric import ErgodicResult, summarize_metrics
from .map_utils import Workspace


def compute_metrics(
    trajectories: np.ndarray,
    controls: np.ndarray,
    q_grid: np.ndarray,
    workspace: Workspace,
    sdf: np.ndarray,
    dt: float,
    fourier_K: int,
    overlap_sigma: float,
) -> Tuple[Dict[str, float], ErgodicResult]:
    """Compute all scalar metrics for a run."""
    return summarize_metrics(
        trajectories=trajectories,
        controls=controls,
        q_grid=q_grid,
        workspace=workspace,
        sdf=sdf,
        dt=dt,
        fourier_K=fourier_K,
        overlap_sigma=overlap_sigma,
    )


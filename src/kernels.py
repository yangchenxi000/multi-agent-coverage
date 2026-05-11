"""Kernel functions used by Stein operators."""

from __future__ import annotations

import numpy as np


def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances."""
    Xn = np.sum(X**2, axis=1, keepdims=True)
    Yn = np.sum(Y**2, axis=1, keepdims=True).T
    return np.maximum(Xn + Yn - 2.0 * (X @ Y.T), 0.0)


def rbf_kernel(X: np.ndarray, Y: np.ndarray, h: float) -> np.ndarray:
    """RBF kernel matrix K_ij = exp(-||x_i-y_j||^2/h)."""
    d2 = pairwise_sq_dists(X, Y)
    return np.exp(-d2 / max(h, 1e-8))


def grad_x_rbf_kernel(X: np.ndarray, Y: np.ndarray, h: float) -> np.ndarray:
    """Gradient wrt first arg x_i of k(x_i, y_j).

    Returns tensor shape (n_x, n_y, dim).
    """
    K = rbf_kernel(X, Y, h)
    diff = X[:, None, :] - Y[None, :, :]
    return (-2.0 / max(h, 1e-8)) * K[:, :, None] * diff


def median_heuristic_bandwidth(X: np.ndarray, floor: float = 1e-3) -> float:
    """Median heuristic for RBF bandwidth."""
    if X.shape[0] < 2:
        return 0.1
    d2 = pairwise_sq_dists(X, X)
    vals = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(vals))
    return max(med + 1e-12, floor)

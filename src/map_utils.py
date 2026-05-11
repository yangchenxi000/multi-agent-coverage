"""Workspace and obstacle map utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Workspace:
    """2D workspace represented by regular grid."""

    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    resolution: int

    @property
    def dx(self) -> float:
        return (self.xlim[1] - self.xlim[0]) / (self.resolution - 1)

    @property
    def dy(self) -> float:
        return (self.ylim[1] - self.ylim[0]) / (self.resolution - 1)

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
        y = np.linspace(self.ylim[0], self.ylim[1], self.resolution)
        return np.meshgrid(x, y, indexing="xy")


def add_rect_obstacle(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, rect: Dict[str, float]) -> None:
    """Mark rectangular obstacle on mask."""
    x0, x1 = rect["x0"], rect["x1"]
    y0, y1 = rect["y0"], rect["y1"]
    inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    mask[inside] = 1


def add_circle_obstacle(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, circle: Dict[str, float]) -> None:
    """Mark circular obstacle on mask."""
    cx, cy, r = circle["cx"], circle["cy"], circle["r"]
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= r**2
    mask[inside] = 1


def add_polygon_obstacle(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, poly: Dict) -> None:
    """Mark polygon obstacle on mask via ray casting."""
    vertices = np.asarray(poly["vertices"], dtype=float)
    px = X.ravel()
    py = Y.ravel()
    inside = np.zeros(px.shape, dtype=bool)

    x1 = vertices[:, 0]
    y1 = vertices[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)

    # Crossing number test for all points against each edge.
    for ex1, ey1, ex2, ey2 in zip(x1, y1, x2, y2):
        cond_y = (ey1 > py) != (ey2 > py)
        x_intersect = (ex2 - ex1) * (py - ey1) / (ey2 - ey1 + 1e-12) + ex1
        inside ^= cond_y & (px < x_intersect)

    mask.ravel()[inside] = 1


def build_obstacle_mask(workspace: Workspace, obstacles: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create binary obstacle mask from obstacle specs."""
    X, Y = workspace.meshgrid()
    mask = np.zeros_like(X, dtype=np.uint8)
    for obs in obstacles:
        if obs["type"] == "rect":
            add_rect_obstacle(mask, X, Y, obs)
        elif obs["type"] == "circle":
            add_circle_obstacle(mask, X, Y, obs)
        elif obs["type"] == "polygon":
            add_polygon_obstacle(mask, X, Y, obs)
        else:
            raise ValueError(f"Unknown obstacle type: {obs['type']}")
    return X, Y, mask


def default_scenarios() -> Dict[str, Dict]:
    """Built-in scenario specifications."""
    return {
        "single_hotspot_open": {
            "hotspots": [
                {"mean": [0.75, 0.75], "cov": [[0.02, 0.0], [0.0, 0.02]], "weight": 1.0}
            ],
            "obstacles": [],
        },
        "single_hotspot_few_obstacles": {
            "hotspots": [
                {"mean": [0.75, 0.75], "cov": [[0.02, 0.0], [0.0, 0.02]], "weight": 1.0}
            ],
            "obstacles": [
                {"type": "circle", "cx": 0.45, "cy": 0.55, "r": 0.12},
                {"type": "rect", "x0": 0.1, "x1": 0.2, "y0": 0.65, "y1": 0.9},
            ],
        },
        "three_hotspots_several_obstacles": {
            "hotspots": [
                {"mean": [0.2, 0.25], "cov": [[0.01, 0.0], [0.0, 0.02]], "weight": 0.35},
                {"mean": [0.78, 0.22], "cov": [[0.015, 0.0], [0.0, 0.015]], "weight": 0.3},
                {"mean": [0.62, 0.78], "cov": [[0.02, 0.0], [0.0, 0.012]], "weight": 0.35},
            ],
            "obstacles": [
                {"type": "rect", "x0": 0.33, "x1": 0.43, "y0": 0.05, "y1": 0.45},
                {"type": "rect", "x0": 0.55, "x1": 0.65, "y0": 0.55, "y1": 0.95},
                {"type": "circle", "cx": 0.5, "cy": 0.5, "r": 0.08},
            ],
        },
        "asymmetric_narrow_passage": {
            "hotspots": [
                {"mean": [0.15, 0.8], "cov": [[0.01, 0.0], [0.0, 0.015]], "weight": 0.55},
                {"mean": [0.82, 0.2], "cov": [[0.01, 0.0], [0.0, 0.01]], "weight": 0.45},
            ],
            "obstacles": [
                {"type": "rect", "x0": 0.35, "x1": 0.6, "y0": 0.0, "y1": 0.42},
                {"type": "rect", "x0": 0.35, "x1": 0.6, "y0": 0.58, "y1": 1.0},
                {"type": "circle", "cx": 0.66, "cy": 0.52, "r": 0.08},
            ],
        },
        "user_custom_scene": {
            "hotspots": [
                {"mean": [0.34, 0.50], "cov": [[0.012, 0.0], [0.0, 0.01]], "weight": 0.33},
                {"mean": [0.64, 0.58], "cov": [[0.012, 0.0], [0.0, 0.018]], "weight": 0.34},
                {"mean": [0.53, 0.25], "cov": [[0.02, 0.0], [0.0, 0.01]], "weight": 0.33},
            ],
            "obstacles": [
                {"type": "rect", "x0": 0.455, "x1": 0.575, "y0": 0.645, "y1": 0.775},
                {"type": "circle", "cx": 0.41, "cy": 0.43, "r": 0.062},
                {
                    "type": "polygon",
                    "vertices": [
                        [0.553, 0.225],
                        [0.595, 0.225],
                        [0.620, 0.270],
                        [0.595, 0.325],
                        [0.545, 0.325],
                        [0.520, 0.270],
                    ],
                },
            ],
        },
    }

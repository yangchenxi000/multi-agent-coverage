"""Mars DEM preprocessing utilities for auto scene construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter, label, maximum_filter, zoom

from .map_utils import Workspace


@dataclass
class MarsSceneData:
    """Container for auto-generated Mars scene."""

    dem_patch: np.ndarray
    base_density: np.ndarray
    obstacle_mask: np.ndarray
    metadata: Dict[str, float]


def _robust_normalize(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Normalize array using robust percentiles."""
    p0, p1 = np.percentile(arr, [lo, hi])
    z = (arr - p0) / max(p1 - p0, 1e-8)
    return np.clip(z, 0.0, 1.0)


def _pick_auto_crop(
    dem: np.ndarray,
    target_size: int = 720,
    complexity: str = "medium",
) -> Tuple[int, int, int]:
    """Pick a deterministic crop with selected terrain complexity."""
    h, w = dem.shape
    size = int(min(target_size, h, w))
    step = max(64, size // 5)
    valid = np.isfinite(dem) & (dem > -20000.0) & (dem < 30000.0)

    candidates = []
    for y0 in range(0, h - size + 1, step):
        for x0 in range(0, w - size + 1, step):
            patch = dem[y0 : y0 + size, x0 : x0 + size]
            vpatch = valid[y0 : y0 + size, x0 : x0 + size]
            valid_ratio = float(np.mean(vpatch))
            if valid_ratio < 0.995:
                continue
            p = patch[vpatch]
            # Prefer medium/complex terrain rather than flat or extreme regions.
            std = float(np.std(p))
            rng = float(np.percentile(p, 95.0) - np.percentile(p, 5.0))
            score = std + 0.2 * rng
            candidates.append((score, y0, x0, size))

    if not candidates:
        return (0, 0, size)

    candidates.sort(key=lambda z: z[0])
    if complexity == "easy":
        idx = int(0.2 * (len(candidates) - 1))
    elif complexity == "hard":
        idx = int(0.85 * (len(candidates) - 1))
    else:
        idx = int(0.55 * (len(candidates) - 1))
    _, y0, x0, size = candidates[idx]
    return (y0, x0, size)


def _make_hotspot_mix(
    q_raw: np.ndarray,
    obstacle_mask: np.ndarray,
    workspace: Workspace,
    max_hotspots: int = 3,
    hotspot_sigma: float = 0.06,
    hotspot_blend: float = 0.3,
) -> np.ndarray:
    """Add sparse Gaussian hotspots on top of terrain-based resource prior."""
    free = np.where(obstacle_mask == 0, q_raw, 0.0)
    peaks = maximum_filter(free, size=15)
    cand = np.argwhere((free == peaks) & (free > np.percentile(free[free > 0], 85.0)))
    if cand.size == 0:
        return q_raw

    cand_vals = free[cand[:, 0], cand[:, 1]]
    order = np.argsort(cand_vals)[::-1]
    selected = []
    min_dist = 24
    for idx in order:
        yx = cand[idx]
        if all(np.linalg.norm(yx - s) >= min_dist for s in selected):
            selected.append(yx)
        if len(selected) >= int(max_hotspots):
            break

    X, Y = workspace.meshgrid()
    q_hot = np.zeros_like(q_raw, dtype=float)
    for y, x in selected:
        cx = float(X[y, x])
        cy = float(Y[y, x])
        sx = float(hotspot_sigma)
        sy = float(hotspot_sigma)
        q_hot += np.exp(-0.5 * (((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))
    if np.max(q_hot) > 0:
        q_hot = q_hot / np.max(q_hot)
    blend = float(np.clip(hotspot_blend, 0.0, 1.0))
    return (1.0 - blend) * q_raw + blend * q_hot


def build_mars_scene_from_dem(
    dem_path: str,
    workspace: Workspace,
    crop_size: int = 720,
    slope_percentile: float = 88.0,
    obstacle_dilation: int = 1,
    complexity: str = "medium",
    crop_y0: int | None = None,
    crop_x0: int | None = None,
    obstacle_opening: int = 0,
    min_component_pixels: int = 0,
    density_smooth_sigma: float = 2.0,
    hotspot_count: int = 2,
    hotspot_sigma: float = 0.08,
    hotspot_blend: float = 0.45,
) -> MarsSceneData:
    """Construct resource density and obstacle mask from Mars DEM."""
    dem = iio.imread(dem_path).astype(np.float64)
    h, w = dem.shape
    size = int(min(crop_size, h, w))
    if crop_y0 is not None and crop_x0 is not None:
        y0 = int(np.clip(crop_y0, 0, h - size))
        x0 = int(np.clip(crop_x0, 0, w - size))
    else:
        y0, x0, size = _pick_auto_crop(dem, target_size=crop_size, complexity=complexity)
    patch = dem[y0 : y0 + size, x0 : x0 + size]
    patch = np.where(np.isfinite(patch), patch, np.nanmedian(patch))

    # Resample to workspace resolution.
    zoom_y = workspace.resolution / patch.shape[0]
    zoom_x = workspace.resolution / patch.shape[1]
    patch_res = zoom(patch, (zoom_y, zoom_x), order=1)
    patch_res = gaussian_filter(patch_res, sigma=1.0)

    h_norm = _robust_normalize(patch_res, 3.0, 97.0)
    gy, gx = np.gradient(h_norm, workspace.dy, workspace.dx)
    slope = np.sqrt(gx**2 + gy**2)
    slope_norm = _robust_normalize(slope, 5.0, 99.0)
    rough = np.abs(np.gradient(gx, axis=1) + np.gradient(gy, axis=0))
    rough_norm = _robust_normalize(rough, 5.0, 99.0)

    obstacle_score = 0.75 * slope_norm + 0.25 * rough_norm
    thr = float(np.percentile(obstacle_score, slope_percentile))
    obstacle_mask = obstacle_score >= thr
    if obstacle_opening > 0:
        st = np.ones((int(obstacle_opening), int(obstacle_opening)), dtype=bool)
        obstacle_mask = binary_opening(obstacle_mask, structure=st)
    if min_component_pixels > 0:
        lb, nlab = label(obstacle_mask)
        if nlab > 0:
            counts = np.bincount(lb.ravel())
            keep = np.zeros_like(counts, dtype=bool)
            keep[1:] = counts[1:] >= int(min_component_pixels)
            obstacle_mask = keep[lb]
    if obstacle_dilation > 0:
        obstacle_mask = binary_dilation(obstacle_mask, iterations=int(obstacle_dilation))
    obstacle_mask = obstacle_mask.astype(np.uint8)

    # Terrain-derived resource prior: low slope + medium elevation band.
    q_raw = np.exp(-((h_norm - 0.52) ** 2) / (2.0 * 0.18**2)) * np.exp(-4.0 * slope_norm)
    q_raw = gaussian_filter(q_raw, sigma=max(float(density_smooth_sigma), 0.0))
    q_mix = _make_hotspot_mix(
        q_raw,
        obstacle_mask,
        workspace,
        max_hotspots=int(hotspot_count),
        hotspot_sigma=float(hotspot_sigma),
        hotspot_blend=float(hotspot_blend),
    )
    q_mix = np.where(obstacle_mask > 0, 0.2 * q_mix, q_mix)

    return MarsSceneData(
        dem_patch=patch_res,
        base_density=q_mix,
        obstacle_mask=obstacle_mask,
        metadata={
            "crop_y0": float(y0),
            "crop_x0": float(x0),
            "crop_size": float(size),
            "slope_percentile": float(slope_percentile),
            "obstacle_ratio": float(np.mean(obstacle_mask)),
        },
    )

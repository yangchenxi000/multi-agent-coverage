"""Visualization utilities for reproduction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np

from .map_utils import Workspace
from .utils import ensure_dir


plt.rcParams.update(
    {
        "figure.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)


def _prep_axes(workspace: Workspace):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(*workspace.xlim)
    ax.set_ylim(*workspace.ylim)
    ax.set_aspect("equal")
    return fig, ax


def plot_heatmap(data: np.ndarray, workspace: Workspace, title: str, save_path: Path, cmap: str = "viridis") -> None:
    """Generic heatmap plot."""
    ensure_dir(save_path.parent)
    fig, ax = _prep_axes(workspace)
    im = ax.imshow(
        data,
        origin="lower",
        extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
        cmap=cmap,
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_combined_map(q: np.ndarray, obstacle_mask: np.ndarray, workspace: Workspace, save_path: Path) -> None:
    """Density plus obstacle overlay."""
    ensure_dir(save_path.parent)
    fig, ax = _prep_axes(workspace)
    ax.imshow(
        q,
        origin="lower",
        extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
        cmap="magma",
        alpha=0.85,
    )
    ax.contour(
        obstacle_mask,
        levels=[0.5],
        origin="lower",
        extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
        colors="cyan",
        linewidths=1.4,
    )
    ax.set_title("Task Density + Obstacles")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_vector_field(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    workspace: Workspace,
    title: str,
    save_path: Path,
    background: Optional[np.ndarray] = None,
) -> None:
    """Quiver vector field visualization."""
    ensure_dir(save_path.parent)
    fig, ax = _prep_axes(workspace)
    if background is not None:
        ax.imshow(
            background,
            origin="lower",
            extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
            cmap="Greys",
            alpha=0.35,
        )
    stride = max(1, X.shape[0] // 24)
    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        U[::stride, ::stride],
        V[::stride, ::stride],
        color="tab:blue",
        alpha=0.85,
        scale=30,
        width=0.003,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_trajectories(
    trajectories: np.ndarray,
    workspace: Workspace,
    save_path: Path,
    density: Optional[np.ndarray] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    title: str = "Trajectories",
) -> None:
    """Plot multi-agent trajectories over optional background density."""
    ensure_dir(save_path.parent)
    fig, ax = _prep_axes(workspace)
    if density is not None:
        ax.imshow(
            density,
            origin="lower",
            extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
            cmap="magma",
            alpha=0.55,
        )
    if obstacle_mask is not None:
        ax.contour(
            obstacle_mask,
            levels=[0.5],
            origin="lower",
            extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
            colors="k",
            linewidths=1.2,
        )

    for i in range(trajectories.shape[0]):
        traj = trajectories[i]
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.6, label=f"agent {i}")
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=28)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=35)
    ax.legend(loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_trajectory_point_vectors(
    trajectories: np.ndarray,
    vectors: np.ndarray,
    workspace: Workspace,
    save_path: Path,
    title: str,
    density: Optional[np.ndarray] = None,
    obstacle_mask: Optional[np.ndarray] = None,
) -> None:
    """Plot arrows on each trajectory point to show Stein direction."""
    ensure_dir(save_path.parent)
    fig, ax = _prep_axes(workspace)
    if density is not None:
        ax.imshow(
            density,
            origin="lower",
            extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
            cmap="magma",
            alpha=0.5,
        )
    if obstacle_mask is not None:
        ax.contour(
            obstacle_mask,
            levels=[0.5],
            origin="lower",
            extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
            colors="k",
            linewidths=1.2,
        )

    for i in range(trajectories.shape[0]):
        traj = trajectories[i]
        vec = vectors[i]
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.4, alpha=0.9)
        ax.quiver(
            traj[:, 0],
            traj[:, 1],
            vec[:, 0],
            vec[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0 / 0.02,
            width=0.003,
            alpha=0.9,
        )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_curves(curves: Dict[str, Iterable[float]], ylabel: str, title: str, save_path: Path) -> None:
    """Plot metric curves."""
    ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, ys in curves.items():
        ax.plot(list(ys), label=name, linewidth=1.7)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_comparison_panel(
    traj_base: np.ndarray,
    traj_ours: np.ndarray,
    density: np.ndarray,
    obstacle_mask: np.ndarray,
    workspace: Workspace,
    save_path: Path,
) -> None:
    """Baseline vs ours two-panel figure."""
    ensure_dir(save_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, traj, t in zip(axes, [traj_base, traj_ours], ["Baseline", "Ours"]):
        ax.imshow(
            density,
            origin="lower",
            extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
            cmap="magma",
            alpha=0.55,
        )
        ax.contour(
            obstacle_mask,
            levels=[0.5],
            origin="lower",
            extent=[workspace.xlim[0], workspace.xlim[1], workspace.ylim[0], workspace.ylim[1]],
            colors="k",
            linewidths=1.2,
        )
        for i in range(traj.shape[0]):
            ax.plot(traj[i, :, 0], traj[i, :, 1], linewidth=1.4)
        ax.set_xlim(*workspace.xlim)
        ax.set_ylim(*workspace.ylim)
        ax.set_aspect("equal")
        ax.set_title(t)
    fig.savefig(save_path)
    plt.close(fig)


def make_trajectory_gif(
    trajectories_hist: List[np.ndarray],
    workspace: Workspace,
    save_path: Path,
    density: Optional[np.ndarray] = None,
) -> None:
    """Generate GIF of trajectory evolution over iterations."""
    ensure_dir(save_path.parent)
    frames: List[np.ndarray] = []
    tmp_dir = save_path.parent / "_gif_tmp"
    ensure_dir(tmp_dir)

    for idx, traj in enumerate(trajectories_hist):
        frame_path = tmp_dir / f"frame_{idx:04d}.png"
        plot_trajectories(traj, workspace, frame_path, density=density, title=f"Iteration snapshot {idx}")
        frames.append(imageio.v2.imread(frame_path))

    if frames:
        imageio.mimsave(save_path, frames, fps=2)

    for p in tmp_dir.glob("*.png"):
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()

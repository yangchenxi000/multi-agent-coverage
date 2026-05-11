"""Common utilities for experiment management."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> np.random.Generator:
    """Set numpy global seed and return a default generator."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def to_serializable(obj: Any) -> Any:
    """Convert objects for JSON serialization."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Save dictionary to YAML."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save dictionary to JSON."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)


class Timer:
    """Simple wall-time timer."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Elapsed seconds."""
        return time.perf_counter() - self._start

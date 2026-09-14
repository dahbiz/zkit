"""Shared utilities for zkit."""

from __future__ import annotations

from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* if it does not exist, and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def discover_runs(root: Union[str, Path]) -> list[Path]:
    """Find all ``TimeEvolutionData_*.h5`` files under *root*."""
    root = Path(root)
    return sorted((root / "td").glob("TimeEvolutionData_*.h5")) if (root / "td").is_dir() else []


__all__ = ["ensure_dir", "discover_runs"]

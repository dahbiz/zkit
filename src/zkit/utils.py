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
    """Time-series files (``td/<name>_observables.h5``, or the former
    ``td/<name>_observables.h5``) under *root*."""
    from zkit.paths import list_runs, output_path

    return [output_path(root, name, "observables") for name in list_runs(root)]


__all__ = ["ensure_dir", "discover_runs"]

"""Base HDF5 utilities for zkit."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np


def _ensure_path(path: Union[str, Path]) -> Path:
    return Path(path)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Expected a file, got directory: {path}")


def list_datasets(path: Union[str, Path]) -> List[str]:
    """List top-level datasets inside an HDF5 file."""
    h5_path = _ensure_path(path)
    _require_file(h5_path)
    with h5py.File(h5_path, "r") as f:
        return [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]


def dataset_shape(path: Union[str, Path], name: str) -> Tuple[int, ...]:
    """Return the shape of a top-level dataset."""
    h5_path = _ensure_path(path)
    _require_file(h5_path)
    with h5py.File(h5_path, "r") as f:
        dset = f[name]
        if not isinstance(dset, h5py.Dataset):
            raise TypeError(f"{path}:{name} is not a dataset")
        return tuple(int(dim) for dim in dset.shape)


def read_dataset(path: Union[str, Path], name: str, *, sl: Optional[slice] = None) -> np.ndarray:
    """Read a top-level dataset as a NumPy array."""
    h5_path = _ensure_path(path)
    _require_file(h5_path)
    with h5py.File(h5_path, "r") as f:
        dset = f[name]
        if not isinstance(dset, h5py.Dataset):
            raise TypeError(f"{path}:{name} is not a dataset")
        arr = np.array(dset) if sl is None else dset[sl]
        return arr


# Dipole dataset widths written by the TDSEZ monitor, per spatial dimension.
_DIPOLE_WIDTH_TO_DIM = {4: 1, 7: 2, 10: 3}


def detect_dimension(path: Union[str, Path]) -> int:
    """Infer the spatial dimension (1, 2, or 3) of a TDSEZ HDF5 output file.

    Works for every TDSEZ output flavour without needing the input file:

      * ``td/TimeEvolutionData_*.h5`` -> dipole column width (4/7/10).
      * ``static/EigenData_*.h5``      -> ``run_metadata`` ``Dimension``
        attribute, falling back to the number of ``knots_*`` datasets.
      * ``td/wfs_*.h5``                -> number of ``knots_*`` datasets.

    Raises
    ------
    ValueError
        If no dimension signal can be found in the file.
    """
    h5_path = _ensure_path(path)
    _require_file(h5_path)
    with h5py.File(h5_path, "r") as f:
        # 1) explicit attribute written by the binary (static eigen output)
        if "run_metadata" in f and "Dimension" in f["run_metadata"].attrs:
            return int(f["run_metadata"].attrs["Dimension"])
        # 2) dipole column width (time-evolution output)
        if "dipoles" in f:
            width = int(f["dipoles"].shape[1])
            if width in _DIPOLE_WIDTH_TO_DIM:
                return _DIPOLE_WIDTH_TO_DIM[width]
        # 3) number of spatial knot vectors (wavefunction / eigen output)
        knots = sum(1 for axis_name in ("x", "y", "z") if f"knots_{axis_name}" in f)
        if knots:
            return knots
    raise ValueError(f"Could not infer simulation dimension from {h5_path}")


__all__ = ["list_datasets", "dataset_shape", "read_dataset", "detect_dimension"]

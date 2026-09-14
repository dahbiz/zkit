"""Read TimeEvolutionData_<inp>.h5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import h5py

from zkit.io.base import read_dataset
from zkit.models import TimeEvolution

_DATASETS = ("dipoles", "populations", "energies", "currents", "autocorrelation")


def read_evolution(path: Union[str, Path], *, dimension: Optional[int] = None) -> TimeEvolution:
    """Read a TimeEvolutionData HDF5 file.

    Parameters
    ----------
    path:
        Path to the HDF5 file.
    dimension:
        1, 2, or 3. When omitted, it is inferred from the file
        automatically (dipole column width), so the input file is never
        required.

    Returns
    -------
    TimeEvolution
    """
    h5_path = Path(path)
    for name in _DATASETS:
        if name not in _list_top_datasets(h5_path):
            raise ValueError(f"Missing expected dataset {name!r} in {h5_path}")

    dipoles = read_dataset(h5_path, "dipoles")
    populations = read_dataset(h5_path, "populations")
    energies = read_dataset(h5_path, "energies")
    currents = read_dataset(h5_path, "currents")
    autocorrelation = read_dataset(h5_path, "autocorrelation")

    if dimension is None:
        # Infer from the dipole column width written by the binary.
        n_dipole_cols = int(dipoles.shape[1])
        mapping = {4: 1, 7: 2, 10: 3}
        if n_dipole_cols not in mapping:
            raise ValueError(f"Unexpected dipole width {n_dipole_cols}; expected 4, 7, or 10")
        dimension = mapping[n_dipole_cols]

    if energies.shape[1] != 7:
        raise ValueError(f"energies must have 7 columns, got {energies.shape[1]}")
    if currents.shape[1] not in (11, 15):
        raise ValueError(
            f"currents must have 11 columns (1D/2D) or 15 columns (3D), got {currents.shape[1]}"
        )
    if autocorrelation.shape[1] != 3:
        raise ValueError(f"autocorrelation must have 3 columns, got {autocorrelation.shape[1]}")

    return TimeEvolution(
        time=dipoles[:, 0].astype(float),
        dipoles=dipoles.astype(float),
        populations=populations.astype(float),
        energies=energies.astype(float),
        currents=currents.astype(float),
        autocorrelation=autocorrelation.astype(float),
        dimension=int(dimension),
    )


def _list_top_datasets(path: Path) -> list:
    with h5py.File(path, "r") as f:
        return [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]


__all__ = ["read_evolution"]

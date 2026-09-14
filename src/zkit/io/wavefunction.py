"""Read td/wfs_<inp>.h5 files produced by PETSc HDF5 viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from zkit.io.base import detect_dimension
from zkit.models import WavefunctionSeries


def read_wfs(path: Union[str, Path]) -> WavefunctionSeries:
    """Read wavefunction snapshot file written by PETSc HDF5 viewer.

    PETSc writes either a group ``wavefunction`` containing numbered
    datasets ``0``, ``1``, ... or a single dataset with an extended
    first dimension. This reader handles both. The spatial dimension is
    inferred from the file (number of ``knots_*`` datasets), so the input
    file is not required.
    """
    h5_path = Path(path)
    with h5py.File(h5_path, "r") as h5_file:
        snapshots: list[np.ndarray] = []

        def _complex_from(wave_obj: h5py.Dataset) -> list[np.ndarray]:
            """Convert an HDF5 dataset (any spatial dimensionality) whose
            trailing dimension is [real, imag] into a list of flattened
            complex per-snapshot vectors.

            PETSc writes the snapshot index as the FIRST axis and the
            real/imaginary pair as the LAST axis, with arbitrary middle
            (spatial-DOF) axes in between:
                1D: (n_snap, n_dof, 2)
                2D: (n_snap, nx, ny, 2)
                3D: (n_snap, nx, ny, nz, 2)
            """
            wave_array = np.array(wave_obj)  # shape (..., 2)
            if wave_array.shape[-1] != 2:
                # real-only data: flatten the leading (snapshot) axis
                return [
                    wave_array[snapshot_index].ravel()
                    for snapshot_index in range(wave_array.shape[0])
                ]
            complex_wave = wave_array[..., 0] + 1j * wave_array[..., 1]  # drop trailing 2-axis
            return [
                complex_wave[snapshot_index].ravel()
                for snapshot_index in range(complex_wave.shape[0])
            ]

        if "wavefunction" in h5_file and isinstance(h5_file["wavefunction"], h5py.Group):
            wave_group = h5_file["wavefunction"]
            keys = sorted(wave_group.keys(), key=lambda k: int(k))
            for snapshot_key in keys:
                dset = wave_group[snapshot_key]
                if isinstance(dset, h5py.Dataset):
                    snapshots.append(np.array(dset).ravel())
        else:
            for obj_key in h5_file.keys():
                obj = h5_file[obj_key]
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] == 2:
                    # PETSc timestep layout: (n_snap, <spatial...>, 2)
                    snapshots.extend(_complex_from(obj))
                    break
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                    # older/real-only layout: first dim = snapshots
                    n_snapshots = int(obj.shape[0])
                    for snapshot_index in range(n_snapshots):
                        snapshots.append(np.array(obj[snapshot_index]).ravel())
                    break

        if not snapshots:
            raise ValueError(f"No wavefunction data found in {h5_path}")

        spatial_dim = detect_dimension(h5_path)
        data = np.vstack(snapshots) if snapshots else np.zeros((0, 0))
        times = np.arange(data.shape[0], dtype=float)

    return WavefunctionSeries(
        times=times, data=data, n_snapshots=data.shape[0], dimension=spatial_dim
    )


__all__ = ["read_wfs"]

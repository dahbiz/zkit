"""Read ts_<inp>.h5 files produced by PETSc TS viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from zkit.models import TSSeries


def read_timeseries(path: Union[str, Path]) -> TSSeries:
    """Read PETSc TS snapshot file.

    PETSc TSView typically writes a group ``timestepper`` containing
    numbered snapshots or a flat layout. This reader returns the
    available step indices as ``times``.
    """
    h5_path = Path(path)
    with h5py.File(h5_path, "r") as h5_file:
        indices: list[int] = []

        if "timestepper" in h5_file and isinstance(h5_file["timestepper"], h5py.Group):
            step_group = h5_file["timestepper"]
            for key in step_group.keys():
                try:
                    indices.append(int(key))
                except ValueError:
                    continue
        else:
            for key in h5_file.keys():
                try:
                    idx = int(key)
                    if idx not in indices:
                        indices.append(idx)
                except ValueError:
                    continue

        if not indices:
            times = np.zeros(0, dtype=float)
        else:
            times = np.array(sorted(indices), dtype=float)

    return TSSeries(times=times, n_snapshots=int(times.shape[0]))


__all__ = ["read_timeseries"]

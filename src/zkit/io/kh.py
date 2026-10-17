"""Mode = kh quiver and multipole metadata from static/<name>_eigen.h5."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from zkit.models import KHQuiver, MultipoleInfo


def read_kh(path: Union[str, Path]) -> KHQuiver:
    """Quiver nodes, weights and displacements written by a ``Mode = kh`` run."""
    with h5py.File(path, "r") as f:
        g = f.get("run_metadata")
        if g is None or "kh_alpha" not in g:
            raise ValueError(f"{path}: no KH quiver (run_metadata/kh_alpha); was it a Mode = kh run?")
        return KHQuiver(
            times=np.asarray(g["kh_times"], dtype=float),
            weights=np.asarray(g["kh_weights"], dtype=float),
            alpha=np.asarray(g["kh_alpha"], dtype=float).reshape(-1, 3),
            window=(float(g.attrs["KH_window_start"]), float(g.attrs["KH_window_end"])),
            omega=float(g.attrs["KH_omega"]),
        )


def read_multipole(path: Union[str, Path]) -> MultipoleInfo:
    """Multipole order, propagation direction, c and magnetic flag (order None if not recorded)."""
    with h5py.File(path, "r") as f:
        a = f["run_metadata"].attrs if "run_metadata" in f else {}
        if "Multipole" not in a:  # written before TDSE-Z recorded it
            return MultipoleInfo(order=None)
        order = int(a["Multipole"])
        if order < 2:
            return MultipoleInfo(order=order)
        return MultipoleInfo(
            order=order,
            direction=tuple(float(a[f"PropagationDirection_{c}"]) for c in "xyz"),
            speed_of_light=float(a["SpeedOfLight"]),
            magnetic=bool(int(a["MultipoleMagnetic"])),
        )

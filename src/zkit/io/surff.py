"""Read td/surff_<inp>.h5 files (t-SURFF photoelectron spectra)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from zkit.models import SurffSpectrum


def read_surff(path: Union[str, Path]) -> SurffSpectrum:
    """Read a t-SURFF file written by TDSE-Z (``TSurff = 1``).

    Parameters
    ----------
    path:
        Path to ``td/surff_<input>.h5``.

    Returns
    -------
    SurffSpectrum
        Momenta, complex amplitudes b(k), |b|^2, the flux part alone (with
        ``SurffInterior`` or ``SurffEndTerm``), the grid axes and ``pes_grid``
        (grid mode) and the file attributes (``final_time``, ``radius``,
        ``A_final``, ``end_term``, ...).
    """
    with h5py.File(Path(path), "r") as f:
        momenta = np.asarray(f["momenta"][()], dtype=float)
        if momenta.ndim == 1:
            momenta = momenta[:, None]
        a = np.asarray(f["amplitude"][()], dtype=float)
        amplitude = a[:, 0] + 1j * a[:, 1]
        pes = np.asarray(f["pes"][()], dtype=float)
        flux = None
        if "amplitude_flux" in f:
            b = np.asarray(f["amplitude_flux"][()], dtype=float)
            flux = b[:, 0] + 1j * b[:, 1]
        axes = tuple(np.asarray(f[n][()], dtype=float) for n in ("k_x", "k_y", "k_z") if n in f)
        pes_grid = np.asarray(f["pes_grid"][()], dtype=float) if "pes_grid" in f else None
        attrs = {}
        for k, v in f.attrs.items():
            v = np.asarray(v)
            attrs[k] = v.item() if v.size == 1 else v
    return SurffSpectrum(
        momenta=momenta,
        amplitude=amplitude,
        pes=pes,
        amplitude_flux=flux,
        axes=axes if axes else None,
        pes_grid=pes_grid,
        attrs=attrs,
    )

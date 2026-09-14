"""Read static/EigenData_<inp>.h5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from zkit.io.base import detect_dimension
from zkit.models import EigenData


def read_eigen(path: Union[str, Path]) -> EigenData:
    """Read eigenvalues and eigenvectors from an EigenData HDF5 file.

    Attempts common PETSc output layouts. The spatial dimension is inferred
    from the file (``run_metadata.Dimension`` attribute, falling back to the
    number of ``knots_*`` datasets), so the input file is not required.

    For TDSEZ static output the per-state eigenvectors are stored as
    top-level datasets ``psi_0, psi_1, ...`` (each ``(n_dof, 2)`` = [Re, Im]),
    which are collected into ``vectors`` as ``(n_states, 2*n_dof)`` so the
    model's 2-D contract holds and ``n_states`` is populated.  The smooth
    real-space wavefunction is then obtained with ``EigenData.reconstruct``
    (which evaluates the B-spline basis via igakit).
    """
    h5_path = Path(path)
    with h5py.File(h5_path, "r") as f:
        if "spectrum" in f:
            spectrum = np.array(f["spectrum"])
            values = spectrum[:, 0].astype(float) if spectrum.ndim == 2 else spectrum.astype(float)
            vectors = None
            for candidate in ("eigenvectors", "vectors", "modes"):
                if candidate in f:
                    vectors = np.array(f[candidate])
                    break
            if vectors is None:
                # TDSEZ static layout: psi_0, psi_1, ... each (n_dof, 2)
                psi_keys = sorted(
                    (k for k in f.keys() if k.startswith("psi_")),
                    key=lambda k: int(k.split("_")[1]),
                )
                if psi_keys:
                    stacked = [
                        np.asarray(f[k], dtype=float).reshape(-1) for k in psi_keys
                    ]  # (n_dof*2,)
                    vectors = np.stack(stacked, axis=0)  # (n_states, n_dof*2)
            if vectors is None:
                # fall back to storing values twice to satisfy the model
                vectors = np.zeros((values.shape[0], 0), dtype=float)
            dimension = detect_dimension(h5_path)
            eigen_data = EigenData(
                values=values, vectors=vectors, n_states=int(vectors.shape[0]), dimension=dimension
            )
            object.__setattr__(eigen_data, "_path", str(h5_path))
            return eigen_data

        # fallback: first 2-D top-level dataset
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                spectrum = np.array(obj)
                values = spectrum[:, 0].astype(float)
                vectors = (
                    spectrum[:, 1:].astype(float)
                    if spectrum.shape[1] > 1
                    else np.zeros((spectrum.shape[0], 0), dtype=float)
                )
                dimension = detect_dimension(h5_path)
                eigen_data = EigenData(
                    values=values,
                    vectors=vectors,
                    n_states=int(vectors.shape[0]),
                    dimension=dimension,
                )
                object.__setattr__(eigen_data, "_path", str(h5_path))
                return eigen_data

    raise ValueError(f"No recognizable eigenvalue data found in {h5_path}")


__all__ = ["read_eigen"]

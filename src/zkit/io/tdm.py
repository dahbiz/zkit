"""Read transition-dipole-moment (TDM) data from TDSEZ static HDF5 output.

The full transition-dipole matrices d_ij = ⟨i|D_α|j⟩ between the saved
eigenstates are written by TDSE-Z's ``dipoles.x`` (run it in the run
directory after a solve with ``NBoundStatesSave = 1`` and ``SaveDipoleAxes``)::

    static/<name>_tdm_Dx.npy, _tdm_Dy.npy, _tdm_Dz.npy   complex (N, N)

:func:`read_tdm` reads them next to ``static/<name>_eigen.h5`` (also under the
former names ``TDM_Dx_<name>.inp.npy``), or the datasets ``tdm_x/y/z``
(shape (N*N, 2), [re, im]) when an eigen file carries them. The text summary
``static/<name>_tdm.txt`` drops small and diagonal entries and is not read.

This module complements the eigen reader
(:func:`zkit.io.eigen.read_eigen`): read the HDF5 EigenData file with either
helper and combine the spectrum/energies with the TDM matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np


def _read_tdm_dataset(h5_file: h5py.File, dataset_name: str) -> Union[np.ndarray, None]:
    """Return the (N, N) complex matrix for ``dataset_name``, or None."""
    if dataset_name not in h5_file:
        return None
    tdm_dset = h5_file[dataset_name]
    flat_real_imag = np.asarray(tdm_dset, dtype=float)  # (N*N, 2) -> [re, im]
    n_states = int(tdm_dset.attrs.get("n_states", 0))
    if n_states == 0:
        # Fall back: infer N from the row count (assumes square matrix).
        n_states = int(round(np.sqrt(flat_real_imag.shape[0])))
    real = flat_real_imag[:, 0]
    imag = flat_real_imag[:, 1]
    return (real + 1j * imag).reshape(n_states, n_states)


def read_tdm(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read the full transition-dipole matrices from a TDSEZ EigenData HDF5.

    Parameters
    ----------
    path
        Path to ``static/<name>_eigen.h5`` (the same file that carries
        ``spectrum`` / ``psi_*`` / ``run_metadata``).

    Returns
    -------
    dict
        Keys ``"tdm_x"``, ``"tdm_y"``, ``"tdm_z"`` (each an
        ``(N, N)`` complex ``np.ndarray`` of ⟨i|D_α|j⟩), present only for the
        spatial axes actually assembled by the run.  If no TDM datasets are
        found, an empty dict is returned (e.g. the file predates this feature).
    """
    h5_path = Path(path)
    if not h5_path.exists():
        raise FileNotFoundError(f"eigen file not found: {h5_path}")
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h5_file:
        for axis_name in ("x", "y", "z"):
            dataset_name = f"tdm_{axis_name}"
            tdm_matrix = _read_tdm_dataset(h5_file, dataset_name)
            if tdm_matrix is not None:
                out[dataset_name] = tdm_matrix
    if out:
        return out
    for axis_name, npy in _tdm_npy_files(h5_path).items():
        out[f"tdm_{axis_name}"] = np.load(npy)
    return out


def _tdm_npy_files(h5_path: Path) -> Dict[str, Path]:
    """dipoles.x output next to an eigen file: static/<name>_tdm_D<a>.npy
    (eigen file <name>_eigen.h5), or TDM_D<a>_<input>.npy (EigenData_<input>.h5)."""
    f = h5_path.name
    if f.endswith("_eigen.h5"):
        name = f[: -len("_eigen.h5")]
        cands = {a: [h5_path.parent / f"{name}_tdm_D{a}.npy"] for a in "xyz"}
    elif f.startswith("EigenData_"):
        inp = f[len("EigenData_") : -len(".h5")]
        stem = inp.rsplit(".", 1)[0]
        cands = {
            a: [h5_path.parent / f"TDM_D{a}_{inp}.npy", h5_path.parent / f"{stem}_tdm_D{a}.npy"]
            for a in "xyz"
        }
    else:
        return {}
    return {
        a: next(p for p in ps if p.exists())
        for a, ps in cands.items()
        if any(p.exists() for p in ps)
    }


def tdm_of_state(
    path: Union[str, Path], source_state: int, axes: Union[str, List[str], None] = None
) -> Dict[str, np.ndarray]:
    """Return the dipole TDM *out of* eigenstate ``source_state`` for every axis.

    The result is the ``source_state``-th row of the requested TDM matrix/matrices, i.e.
    the vector ``[⟨source_state|D_α|0⟩, ⟨source_state|D_α|1⟩, ..., ⟨source_state|D_α|N-1⟩]`` of transition
    dipoles from state ``source_state`` into all other eigenstates.

    Parameters
    ----------
    path
        EigenData HDF5 path (passed to :func:`read_tdm`).
    source_state
        Source eigenstate index (0-based).
    axes
        Which axes to return.  One of ``"x"``, ``"y"``, ``"z"``, a list of
        those, or ``None`` (default → all axes present in the file).

    Returns
    -------
    dict
        Mapping ``axis -> (N,) complex ndarray`` (the row of that TDM matrix).
        If ``axes`` is a single string the dict still uses that string as key.

    Raises
    ------
    IndexError
        If ``source_state`` is out of range for the stored TDM matrices.
    KeyError
        If a requested axis has no TDM dataset in the file.
    """
    tdm = read_tdm(path)
    if not tdm:
        raise KeyError(
            f"no transition dipoles for {path}: run dipoles.x in the run directory "
            f"(after a solve with NBoundStatesSave = 1 and SaveDipoleAxes)"
        )
    if axes is None:
        want = list(tdm.keys())  # e.g. ["tdm_x", "tdm_z"]
    elif isinstance(axes, str):
        want = [f"tdm_{axes}"]
    else:
        want = [f"tdm_{axis_name}" for axis_name in axes]

    n_states = next(iter(tdm.values())).shape[0]
    if not (0 <= source_state < n_states):
        raise IndexError(f"state index {source_state} out of range for N={n_states} states")

    out: Dict[str, np.ndarray] = {}
    for key in want:
        if key not in tdm:
            missing = key.split("_")[-1]
            raise KeyError(f"no TDM dataset for axis '{missing}' in {path}")
        out[key] = tdm[key][
            source_state, :
        ]  # row source_state = transitions FROM state source_state
    return out


def tdm_magnitude_of_state(
    path: Union[str, Path], source_state: int, axes: Union[str, List[str], None] = None
) -> np.ndarray:
    """Convenience: |⟨source_state|D_α|j⟩| (Euclidean norm over the requested axes) per j.

    Returns an ``(N,)`` real array giving the total dipole-transition magnitude
    out of state ``source_state`` into every other state, summed in quadrature over the
    requested axes.
    """
    rows = tdm_of_state(path, source_state, axes=axes)
    sq = np.zeros(next(iter(rows.values())).shape[0], dtype=float)
    for value in rows.values():
        sq += np.abs(value) ** 2
    return np.sqrt(sq)


__all__ = ["read_tdm", "tdm_of_state", "tdm_magnitude_of_state"]

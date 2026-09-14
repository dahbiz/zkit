"""zkit — post-processing toolkit for TDSEZ eigenstate HDF5 outputs.

This package reads the EigenData_*.h5 / wfs_*.h5 files written by the TDSEZ
binary and provides reconstruction + analytic-reference helpers used by the
validation test-suite (tests/test_zkit.py).

HDF5 layout (as written by src/core.cpp :: TDSEZCore::Output):
  /spectrum                (1-D float)  converged eigenvalues, a.u.
  /run_metadata            (group)      echoed input params + code_version
      code_version         (attr str)
      input_file           (attr str)
      units                (attr str)
      Dimension            (attr int)
      SplineDegree         (attr int)
      Nelements            (attr int)
      LMinX..LMaxZ         (attr float)
      Hbar, Charge         (attr float)
      TargetEigenvalue     (attr float)
      NBoundStates*        (attr int)
      eig_residual         (dataset 1-D float)  SLEPc relative error per state
  /knots_x, /knots_y, /knots_z   (1-D float)  PetIGA compact knot vector
                                    len = nfuncs + p ; last knot = Lmax
  /psi_0 .. /psi_{N-1}     (1-D complex)  eigenvector coefficients in IGA dof order
"""

import h5py
import numpy as np

from zkit._version import __version__
from zkit.io import (
    compute_wfs_norm,
    detect_dimension,
    read_eigen,
    read_evolution,
    read_tdm,
    read_timeseries,
    read_wfs,
    reconstruct_static_wfs,
    reconstruct_wfs,
    tdm_magnitude_of_state,
    tdm_of_state,
    wfs_to_vtk,
)
from zkit.models import (
    EigenData,
    SimulationMeta,
    TimeEvolution,
    TSSeries,
    WavefunctionSeries,
)
from zkit.simulation import Run, load
from zkit.viz import plot_tdm, plot_tdm_matrix, plot_transition_diagram, plot_wavefunction

__all__ = [
    "CHARGE",
    "HBAR",
    "EigenData",
    "Run",
    "SimulationMeta",
    "TSSeries",
    "TimeEvolution",
    "WavefunctionSeries",
    "__version__",
    "compute_wfs_norm",
    "cox_de_boor",
    "detect_dimension",
    "find_binary",
    "infinite_well_energy",
    "load",
    "open_eig",
    "partition_of_unity",
    "plot_tdm",
    "plot_tdm_matrix",
    "plot_transition_diagram",
    "plot_wavefunction",
    "read_eigen",
    "read_evolution",
    "read_tdm",
    "read_timeseries",
    "read_wfs",
    "reconstruct_knots",
    "reconstruct_static_wfs",
    "reconstruct_wfs",
    "run",
    "tdm_magnitude_of_state",
    "tdm_of_state",
    "wfs_to_vtk",
]

# Code units (must match TDSEZParser defaults / the HDF5 'units' attribute)
HBAR = 1.0
CHARGE = 1.0
# TDSEZ convention:  hbar^2 / 2m  ==  1/2   (i.e. m = hbar^2 == 1.0 in these units)
# => infinite well  E_n = (pi^2 / 2) * n^2 / L^2


def open_eig(fname):
    """Return a dict with spectrum, metadata, knots, and (if present) psi vectors."""
    out = {"file": fname}
    with h5py.File(fname, "r") as f:
        spec = np.asarray(f["spectrum"][()], dtype=float)
        # The binary writes spectrum as shape (n, 2); the eigenvalues live in
        # column 0. Flatten to a 1-D array of eigenvalues for callers.
        if spec.ndim == 2:
            spec = spec[:, 0]
        out["spectrum"] = spec
        out["has_metadata"] = "run_metadata" in f
        if out["has_metadata"]:
            md = f["run_metadata"]
            out["meta"] = {}
            # provenance scalars are stored as HDF5 ATTRIBUTES on the group
            for k, v in md.attrs.items():
                out["meta"][k] = v
            # (any) datasets under the group, e.g. eig_residual
            for k in md.keys():
                out["meta"][k] = md[k][()]
            # decode byte-string values
            for k, v in list(out["meta"].items()):
                if isinstance(v, bytes):
                    out["meta"][k] = v.decode()
            if "eig_residual" in out["meta"]:
                out["eig_residual"] = np.asarray(out["meta"].pop("eig_residual"), dtype=float)
        for ax in ("x", "y", "z"):
            key = f"knots_{ax}"
            if key in f:
                out[f"knots_{ax}"] = np.asarray(f[key][()], dtype=float)
        # eigenvectors (if saved)
        psis = {}
        i = 0
        while f"psi_{i}" in f:
            psis[i] = np.asarray(f[f"psi_{i}"][()])
            i += 1
        out["psis"] = psis
    return out


def reconstruct_knots(compact, p):
    """Append one final knot equal to the last value (PetIGA compact -> open)."""
    kv = np.asarray(compact, dtype=float)
    return np.append(kv, kv[-1])


def cox_de_boor(x, kv, p, i):
    """Evaluate basis function B_i^p at scalar x via the Cox-de Boor recursion."""
    kv = np.asarray(kv, dtype=float)
    if p == 0:
        return 1.0 if (kv[i] <= x < kv[i + 1]) else 0.0
    denom1 = kv[i + p] - kv[i]
    denom2 = kv[i + p + 1] - kv[i + 1]
    left = (x - kv[i]) / denom1 * cox_de_boor(x, kv, p - 1, i) if denom1 != 0 else 0.0
    right = (kv[i + p + 1] - x) / denom2 * cox_de_boor(x, kv, p - 1, i + 1) if denom2 != 0 else 0.0
    return left + right


def partition_of_unity(kv, p, xs):
    """Sum over all basis functions at each x. Should be 1.0 everywhere inside."""
    kv = np.asarray(kv, dtype=float)
    n = len(kv) - p - 1  # number of basis functions
    xs = np.asarray(xs, dtype=float)
    res = np.zeros_like(xs)
    for i in range(n):
        for j, x in enumerate(xs):
            res[j] += cox_de_boor(x, kv, p, i)
    return res


def infinite_well_energy(n, L, m=1.0):
    """E_n for an infinite square well of width L in TDSEZ units (hbar^2/2m = 1/2)."""
    return (np.pi**2 / 2.0) * (n**2) / (L**2)


def find_binary(prefer="tdsez"):
    """Locate the built tdsez executable under build-split/ (or build/)."""
    import glob
    import os

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("build-split", "build"):
        cands = sorted(glob.glob(os.path.join(here, d, prefer)))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"could not find '{prefer}' binary under {here}")


def run(input_file, binary=None, nproc=4):
    """Run the tdsez binary on input_file (via mpirun) and return the output path."""
    import os
    import subprocess

    if binary is None:
        binary = find_binary()
    base = os.path.basename(input_file)
    out_h5 = os.path.join("static", f"EigenData_{base}.h5")
    cmd = ["mpirun", "-np", str(nproc), binary, "-inp", input_file]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"tdsez exited {r.returncode}\n--- stdout ---\n{r.stdout.decode()[-2000:]}\n"
            f"--- stderr ---\n{r.stderr.decode()[-2000:]}"
        )
    if not os.path.exists(out_h5):
        raise FileNotFoundError(f"expected output {out_h5} not created")
    return out_h5

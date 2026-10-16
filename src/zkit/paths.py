"""Output file names of a TDSE-Z run.

Every file starts with the run name, the input file name without its
extension (``h2p.inp`` -> ``h2p``)::

    static/h2p_eigen.h5        spectrum and eigenstates
    static/h2p_tdm.txt         transition-dipole summary
    static/h2p_Dx.bin          dipole matrix (SaveDipoleAxes)
    td/h2p_observables.h5      time series (dipole, populations, energies, ...)
    td/h2p_wfs.h5              wavefunction snapshots
    td/h2p_surff.h5            t-SURFF spectrum
    td/h2p_ts.h5               time-stepper snapshots

Runs written before used ``static/EigenData_h2p.inp.h5``,
``td/TimeEvolutionData_h2p.inp.h5``, ``td/wfs_h2p.inp.h5``,
``td/surff_h2p.inp.h5`` and ``ts_h2p.inp.h5``; :func:`output_path` falls back
to those names, so both layouts are read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# kind -> (directory, new suffix, legacy directory, legacy prefix, legacy suffix)
_KINDS = {
    "eigen": ("static", "eigen.h5", "static", "EigenData_", ".h5"),
    "tdm": ("static", "tdm.txt", "static", "TDMInfo_", ".txt"),
    "observables": ("td", "observables.h5", "td", "TimeEvolutionData_", ".h5"),
    "wfs": ("td", "wfs.h5", "td", "wfs_", ".h5"),
    "surff": ("td", "surff.h5", "td", "surff_", ".h5"),
    "ts": ("td", "ts.h5", ".", "ts_", ".h5"),
}


def run_name(input_file: PathLike) -> str:
    """Run name: the input file name without directory and extension."""
    p = Path(input_file)
    return p.stem if p.suffix else p.name


def output_path(run_dir: PathLike, input_file: PathLike, kind: str) -> Path:
    """Path of one output of a run: the current name, or the legacy name when
    only that exists (the input given as ``h2p.inp`` or ``h2p``).

    ``kind`` is one of eigen, tdm, observables, wfs, surff, ts.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}")
    d, suffix, ld, lprefix, lsuffix = _KINDS[kind]
    run_dir = Path(run_dir)
    name = run_name(input_file)
    current = run_dir / d / f"{name}_{suffix}"
    if current.exists():
        return current
    for inp in (Path(input_file).name, f"{name}.inp"):
        old = run_dir / ld / f"{lprefix}{inp}{lsuffix}"
        if old.exists():
            return old
    return current


def list_runs(run_dir: PathLike) -> list:
    """Run names with a time series in ``run_dir/td`` (both layouts)."""
    td = Path(run_dir) / "td"
    if not td.is_dir():
        return []
    names = {p.name[: -len("_observables.h5")] for p in td.glob("*_observables.h5")}
    names |= {
        run_name(p.name[len("TimeEvolutionData_") : -len(".h5")])
        for p in td.glob("TimeEvolutionData_*.h5")
    }
    return sorted(names)


__all__ = ["list_runs", "output_path", "run_name"]

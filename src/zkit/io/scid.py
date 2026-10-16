"""Readers for SCID-TDSE output tables, for benchmarking TDSE-Z.

SCID-TDSE (https://github.com/MBI-Theory/scid-tdse) solves the TDSE in
spherical coordinates. Its text tables (optionally gzipped) are read here:

* ``detail_output`` (``detail.table``): time series of the norm, energy,
  dipole, dipole velocity and acceleration;
* ``field_preview`` (``field.table``): vector potential and electric field;
* ``sts_volkov_table`` / ``sts_coulomb_table``: t-SURF amplitudes on a
  (|k|, theta, phi) product grid.

The column positions are taken from each table's own header, not hard-coded.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Union

import numpy as np

from zkit.models import ScidSpectrum

PathLike = Union[str, Path]


def _header_lines(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    lines = []
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            lines.append(line)
    return lines


def _detail_columns(path: Path) -> dict:
    """Map column titles of a detail table ('#  12  345,376   Re[...]') to 0-based indices."""
    cols = {}
    for line in _header_lines(path):
        m = re.match(r"#\s*(\d+)\s+\d+,\d+\s+(.*\S)", line)
        if m:
            cols[m.group(2)] = int(m.group(1)) - 1
    return cols


def read_scid_detail(path: PathLike) -> dict:
    """Time series of a SCID-TDSE ``detail_output`` table.

    Returns a dict of 1-D arrays: ``time``, ``vector_potential``, ``norm``
    (Re <L|R>), ``energy`` (Re <H>, with the CAP), ``dipole`` (n, 3),
    ``velocity`` (n, 3) and ``acceleration`` (n, 3) (real parts, x, y, z;
    the dipole of the electron is ``<e.r>`` as SCID-TDSE writes it) and
    ``field`` (n, 3), the electric field at the same times.
    """
    path = Path(path)
    cols = _detail_columns(path)
    data = np.loadtxt(path, comments="#")

    def col(pattern):
        for title, i in cols.items():
            if re.fullmatch(pattern, title):
                return data[:, i]
        raise KeyError(f"column {pattern!r} not found in {path}")

    out = {
        "time": col(r"Time, au\[t\]"),
        "vector_potential": col(r"Vector-potential magnitude"),
        "norm": col(r"Re\[<Psi_L\|Psi_R>\]"),
        "energy": col(r"Re\[<Psi_L\|H_at\+H_L\+V_cap\|Psi_R>\], Hartree"),
    }
    derivs = (("dipole", r""), ("velocity", r"\(d/d t\) "), ("acceleration", r"\(d\^2/d t\^2\) "))
    for key, pref in derivs:
        out[key] = np.stack(
            [col(rf"Re\[{pref}<Psi_L\|e \. {a}\|Psi_R>\].*") for a in "xyz"], axis=1
        )
    out["field"] = np.stack([col(rf"Electric field, lab {a}.*") for a in "XYZ"], axis=1)
    return out


def read_scid_field(path: PathLike) -> dict:
    """``time``, ``vector_potential`` (magnitude, theta, phi) and ``field`` (n, 3)
    of a SCID-TDSE ``field_preview`` table (integer steps only)."""
    data = np.loadtxt(Path(path), comments="#")
    step = data[:, 0]
    keep = np.isclose(step, np.round(step))
    data = data[keep]
    return {"time": data[:, 1], "vector_potential": data[:, 2:5], "field": data[:, 5:8]}


def read_scid_pes(path: PathLike) -> ScidSpectrum:
    """t-SURF amplitudes of a SCID-TDSE ``sts_volkov_table`` or ``sts_coulomb_table``.

    The table lists A(k) on a product grid of |k| and the direction (theta,
    phi, in degrees). |A(k)|^2 = dP/d^3k (canonical momentum).
    """
    s = np.loadtxt(Path(path), comments="#", usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    k = np.unique(np.round(s[:, 3], 10))
    th = np.unique(np.round(s[:, 4], 6))
    ph = np.unique(np.round(s[:, 5], 6))
    amp = np.zeros((len(k), len(th), len(ph)), dtype=complex)
    ik = np.searchsorted(k, np.round(s[:, 3], 10))
    it = np.searchsorted(th, np.round(s[:, 4], 6))
    ip = np.searchsorted(ph, np.round(s[:, 5], 6))
    amp[ik, it, ip] = s[:, 6] + 1j * s[:, 7]
    return ScidSpectrum(k=k, theta=np.radians(th), phi=np.radians(ph), amplitude=amp)

"""One-to-one comparison of a TDSE-Z run with a SCID-TDSE run.

Both runs must describe the same physics: the same atom, a linearly
polarised field along z, and t-SURF(F) amplitudes in the canonical momentum.
SCID-TDSE outputs are read from its ``detail_output`` and ``sts_volkov_table``
(or ``sts_coulomb_table``) files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from zkit.hhg import harmonic_spectrum, log_ratio
from zkit.io.scid import read_scid_detail, read_scid_pes
from zkit.io.surff import read_surff
from zkit.photoelectron import (
    angular_distribution,
    ati_peaks,
    compare_spectra,
    energy_spectrum,
    to_spherical,
)

PathLike = Union[str, Path]


@dataclass
class ScidComparison:
    """Data and metrics of a TDSE-Z / SCID-TDSE comparison."""

    time: np.ndarray
    field: dict
    dipole: dict
    acceleration: dict
    order: np.ndarray
    harmonics: dict
    k: np.ndarray
    theta: np.ndarray
    energy: np.ndarray
    density: dict  # dP/d^3k on (k, theta)
    dpde: dict
    peaks: list
    metrics: dict = field(default_factory=dict)
    labels: tuple = ("TDSE-Z", "SCID-TDSE")

    def report(self) -> str:
        m, (lz, ls) = self.metrics, self.labels
        out = [
            f"field E_z(t)        max |diff| / max |E| : {m['field']:.1e}",
            f"dipole <z>(t)       relative L2 diff     : {m['dipole']:.2e}",
            f"acceleration (t)    relative L2 diff     : {m['acceleration']:.2e}",
            f"harmonics           median |log10 ratio| : {m['harmonics']:.3f} decades",
            "",
            f"{'photoelectron yield':30s} {lz:>11s} {ls:>11s} {'ratio':>7s}",
        ]
        y = m["yields"]
        for lab in y["ratio"]:
            out.append(
                f"{lab:30s} {y['test'][lab]:11.4e} {y['reference'][lab]:11.4e} "
                f"{y['ratio'][lab]:7.3f}"
            )
        lo, hi = y["pointwise"]
        e_lo, e_hi = y["pointwise_energy"]
        out.append(
            f"point-wise ratio on the first ATI peak (dP/dE > 10 % of max, "
            f"E {e_lo:.2f}-{e_hi:.2f}) : {lo:.3f} .. {hi:.3f}"
        )
        return "\n".join(out)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare_with_scid(
    run_dir: PathLike,
    basename: str,
    scid_detail: PathLike,
    scid_pes: PathLike,
    omega: float,
    e0: float,
    ip: float = 0.5,
    orders: tuple = (2, 3),
    regions: Optional[list] = None,
    max_order: float = 5.5,
) -> ScidComparison:
    """Compare a TDSE-Z run (``run_dir``, input ``basename``) with SCID-TDSE tables.

    The TDSE-Z run must have ``SaveDipoleAxes`` including z and a grid-mode
    t-SURFF file (``TSurff = 1``) holding the (k_x, k_z) plane or a cube. The
    TDSE-Z spectrum is interpolated onto SCID-TDSE's (|k|, theta) grid.
    """
    from zkit.simulation import Run

    run = Run(run_dir, basename)
    ev = run.evolution
    dim = ev.dimension
    tz, dp = ev.time, ev.dipoles
    sc = read_scid_detail(scid_detail)
    t = tz[tz <= min(sc["time"][-1], tz[-1])]

    def on(tt, y):
        return np.interp(t, tt, y)

    ez = {"TDSE-Z": on(tz, dp[:, dim]), "SCID-TDSE": on(sc["time"], sc["field"][:, 2])}
    dz = {"TDSE-Z": on(tz, dp[:, 2 * dim]), "SCID-TDSE": on(sc["time"], sc["dipole"][:, 2])}
    az = {
        "TDSE-Z": on(tz, dp[:, 3 * dim]),
        "SCID-TDSE": on(sc["time"], sc["acceleration"][:, 2]),
    }
    hh = {lab: harmonic_spectrum(t, d, "dipole", omega0=omega) for lab, d in dz.items()}
    order = hh["TDSE-Z"][0]
    harm = {lab: h[1] for lab, h in hh.items()}
    ref = harm["SCID-TDSE"]
    hmask = (order > 0.5) & (order < max_order) & (ref > 1e-9 * ref.max())

    scid = read_scid_pes(scid_pes)
    k, theta = scid.k, scid.theta
    dens = {
        "TDSE-Z": to_spherical(read_surff(run.td_dir / f"surff_{run.base}.h5"), k, theta),
        "SCID-TDSE": scid.density.mean(axis=2),
    }
    energy = k**2 / 2
    dpde = {lab: energy_spectrum(k, theta, d)[1] for lab, d in dens.items()}
    if regions is None:
        regions = [(0.0, energy[-1] + 1e-12, "total")]
    peaks = ati_peaks(omega, ip, e0, orders)
    window = (peaks[0] - 0.3 * omega, peaks[0] + 0.3 * omega)
    metrics = {
        "field": float(
            np.max(np.abs(ez["TDSE-Z"] - ez["SCID-TDSE"])) / np.max(np.abs(ez["SCID-TDSE"]))
        ),
        "dipole": _rel(dz["TDSE-Z"], dz["SCID-TDSE"]),
        "acceleration": _rel(az["TDSE-Z"], az["SCID-TDSE"]),
        "harmonics": log_ratio(harm["TDSE-Z"], ref, hmask),
        "yields": compare_spectra(
            energy, dpde["TDSE-Z"], dpde["SCID-TDSE"], regions, window=window
        ),
    }
    return ScidComparison(
        time=t,
        field=ez,
        dipole=dz,
        acceleration=az,
        order=order,
        harmonics=harm,
        k=k,
        theta=theta,
        energy=energy,
        density=dens,
        dpde=dpde,
        peaks=peaks,
        metrics=metrics,
    )


def plot_comparison(
    cmp: ScidComparison,
    outdir: PathLike,
    surff_file: Optional[PathLike] = None,
    scid_pes: Optional[PathLike] = None,
    prefix: str = "scid",
    dpi: int = 180,
) -> list:
    """Write the time-series, harmonic, dP/dE, angular and (with both files)
    momentum-map figures of a comparison; returns the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    from zkit.viz.photoelectron import (
        plot_angular,
        plot_harmonics,
        plot_momentum_map,
        plot_pes,
        plot_time_series,
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "time": plot_time_series(
            cmp.time,
            [
                (r"$E_z(t)$ (a.u.)", cmp.field),
                (r"$\langle z\rangle$ (a.u.)", cmp.dipole),
                (r"$\ddot d_z$ (a.u.)", cmp.acceleration),
            ],
            title="Field, dipole and dipole acceleration",
        ),
        "harmonics": plot_harmonics(
            cmp.order, cmp.harmonics, title="Harmonic spectrum (from the dipole)"
        ),
        "pes": plot_pes(cmp.energy, cmp.dpde, "SCID-TDSE", cmp.peaks),
    }
    panels = []
    for e in cmp.peaks:
        ref_e, _ = angular_distribution(cmp.energy, cmp.density["SCID-TDSE"], e, False)
        i = int(np.argmin(np.abs(cmp.energy - ref_e)))
        top = cmp.density["SCID-TDSE"][i].max()
        panels.append(
            (f"E = {ref_e:.3f} a.u.", {lab: d[i] / top for lab, d in cmp.density.items()})
        )
    figs["angular"] = plot_angular(cmp.theta, panels)
    if surff_file is not None:
        other = read_scid_pes(scid_pes) if scid_pes is not None else None
        kmax = float(cmp.k[-1]) if other is not None else None
        figs["kmap"] = plot_momentum_map(read_surff(surff_file), other, kmax=kmax)
    paths = []
    for name, fig in figs.items():
        p = out / f"{prefix}_{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths


__all__ = ["ScidComparison", "compare_with_scid", "plot_comparison"]

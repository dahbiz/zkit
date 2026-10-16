"""Photoelectron spectra from t-SURFF amplitudes.

TDSE-Z writes |b(k)|^2 on a Cartesian momentum grid (or a list of momenta);
SCID-TDSE writes A(k) on a spherical (|k|, theta, phi) grid. These helpers put
both on the same spherical grid and reduce them to dP/dE, angular
distributions and yields per energy window. Atomic units throughout, unit
mass outside the surface (E = k^2 / 2).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from zkit.models import ScidSpectrum, SurffSpectrum


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _interp_grid(axes: Sequence[np.ndarray], values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Multilinear interpolation on a product grid; zero outside it."""
    out = np.zeros(len(points))
    inside = np.ones(len(points), dtype=bool)
    lo, frac = [], []
    for d, ax in enumerate(axes):
        x = points[:, d]
        inside &= (x >= ax[0]) & (x <= ax[-1])
        i = np.clip(np.searchsorted(ax, x, side="right") - 1, 0, len(ax) - 2)
        lo.append(i)
        frac.append(np.clip((x - ax[i]) / (ax[i + 1] - ax[i]), 0.0, 1.0))
    for corner in range(1 << len(axes)):
        idx, w = [], np.ones(len(points))
        for d in range(len(axes)):
            hi = (corner >> d) & 1
            idx.append(lo[d] + hi)
            w = w * (frac[d] if hi else 1.0 - frac[d])
        out += w * values[tuple(idx)]
    return np.where(inside, out, 0.0)


def to_spherical(
    spec: SurffSpectrum,
    k: np.ndarray,
    theta: np.ndarray,
    phi: float = 0.0,
) -> np.ndarray:
    """|b(k)|^2 of a TDSE-Z grid spectrum at (|k|, theta) for the azimuth ``phi``.

    The polar axis is z (the last momentum axis). Axes of length one (a
    ``k_y = 0`` plane, say) are dropped: the points are taken in that plane.
    Points outside the momentum grid get zero. Returns an array (n_k, n_theta).
    """
    if not spec.is_grid:
        raise ValueError("to_spherical needs a grid-mode t-SURFF file (SurffNk on every axis)")
    k, theta = np.asarray(k, dtype=float), np.asarray(theta, dtype=float)
    K, T = np.meshgrid(k, theta, indexing="ij")
    dim = len(spec.axes)
    if dim == 1:
        comps = [K * np.cos(T)]
    elif dim == 2:
        comps = [K * np.sin(T), K * np.cos(T)]
    else:
        comps = [K * np.sin(T) * np.cos(phi), K * np.sin(T) * np.sin(phi), K * np.cos(T)]
    grid = spec.pes_grid.reshape([len(a) for a in spec.axes])
    keep = [d for d, a in enumerate(spec.axes) if len(a) > 1]
    axes = [np.asarray(spec.axes[d]) for d in keep]
    values = grid.reshape([len(a) for a in axes])
    pts = np.stack([comps[d].ravel() for d in keep], axis=-1)
    return np.maximum(_interp_grid(axes, values, pts).reshape(K.shape), 0.0)


def _theta_weights(theta: np.ndarray) -> np.ndarray:
    """Cell widths of a theta grid on [0, pi] (pi/n for uniform mid-points)."""
    edges = np.concatenate([[0.0], 0.5 * (theta[1:] + theta[:-1]), [np.pi]])
    return np.diff(edges)


def energy_spectrum(k: np.ndarray, theta: np.ndarray, density: np.ndarray) -> tuple:
    """dP/dE from dP/d^3k on a (|k|, theta) grid, assuming azimuthal symmetry.

    dP/dE = k * 2 pi * int dP/d^3k sin(theta) dtheta. Returns (E, dP/dE).
    """
    k, theta = np.asarray(k, dtype=float), np.asarray(theta, dtype=float)
    w = np.sin(theta) * _theta_weights(theta)
    return k**2 / 2.0, k * 2.0 * np.pi * (np.asarray(density) @ w)


def scid_energy_spectrum(spec: ScidSpectrum) -> tuple:
    """(E, dP/dE) of a SCID-TDSE spectrum, averaged over its phi values."""
    return energy_spectrum(spec.k, spec.theta, spec.density.mean(axis=2))


def angular_distribution(
    energy: np.ndarray, density: np.ndarray, at: float, normalise: bool = True
) -> tuple:
    """dP/d^3k versus theta at the grid energy closest to ``at``.

    Returns (E_used, values); with ``normalise`` the maximum is one.
    """
    i = int(np.argmin(np.abs(np.asarray(energy) - at)))
    y = np.asarray(density)[i].astype(float)
    if normalise and y.max() > 0:
        y = y / y.max()
    return float(energy[i]), y


def region_yields(
    energy: np.ndarray,
    dpde: np.ndarray,
    regions: Iterable[tuple],
) -> dict:
    """Integral of dP/dE over each (lo, hi[, label]) window, keyed by label."""
    out = {}
    for r in regions:
        lo, hi = r[0], r[1]
        label = r[2] if len(r) > 2 else f"{lo:g}-{hi:g}"
        w = (energy >= lo) & (energy < hi)
        out[label] = _trapz(dpde[w], energy[w])
    return out


def ati_peaks(omega: float, ip: float, e0: float, orders: Iterable[int]) -> list:
    """ATI peak energies n*omega - Ip - Up for a field of amplitude ``e0``."""
    up = (e0 / (2.0 * omega)) ** 2
    return [n * omega - ip - up for n in orders]


def compare_spectra(
    energy: np.ndarray,
    test: np.ndarray,
    reference: np.ndarray,
    regions: Optional[Iterable[tuple]] = None,
    peak_fraction: float = 0.1,
    window: Optional[tuple] = None,
) -> dict:
    """Yield ratios test/reference per window and the point-wise ratio range
    where the reference exceeds ``peak_fraction`` of its maximum (inside the
    energy ``window`` (lo, hi) when given; ``pointwise_energy`` is the range used)."""
    regions = list(regions or [(energy[0], energy[-1] + 1e-12, "total")])
    yt = region_yields(energy, test, regions)
    yr = region_yields(energy, reference, regions)
    ratios = {lab: (yt[lab] / yr[lab] if yr[lab] > 0 else np.nan) for lab in yr}
    inside = np.ones(len(energy), dtype=bool)
    if window is not None:
        inside = (energy >= window[0]) & (energy <= window[1])
    m = inside & (reference > peak_fraction * reference[inside].max())
    r = test[m] / reference[m]
    return {
        "test": yt,
        "reference": yr,
        "ratio": ratios,
        "pointwise": (float(r.min()), float(r.max())) if r.size else (np.nan, np.nan),
        "pointwise_energy": (float(energy[m].min()), float(energy[m].max()))
        if r.size
        else (np.nan, np.nan),
    }


__all__ = [
    "angular_distribution",
    "ati_peaks",
    "compare_spectra",
    "energy_spectrum",
    "region_yields",
    "scid_energy_spectrum",
    "to_spherical",
]

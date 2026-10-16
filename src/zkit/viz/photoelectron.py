"""Plots of photoelectron spectra, momentum maps and harmonic spectra.

Every function takes labelled curves ``{label: y}`` so that runs of TDSE-Z and
other codes (SCID-TDSE, ...) can be overlaid; the first label is drawn solid
and the next ones dashed unless a style is given.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from zkit.models import ScidSpectrum, SurffSpectrum

COLORS = ("#1f4e9c", "#d1495b", "#2a9d5c", "#e8a33d", "#6a4c93")
LINES = ("-", "--", "-.", ":", (0, (5, 1)))


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _style(i: int, styles: Optional[dict], label: str) -> dict:
    if styles and label in styles:
        return styles[label]
    return {
        "color": COLORS[i % len(COLORS)],
        "ls": LINES[i % len(LINES)],
        "lw": 1.6 - 0.4 * (i > 0),
    }


def plot_pes(
    energy: np.ndarray,
    spectra: dict,
    reference: Optional[str] = None,
    peaks: Sequence[float] = (),
    decades: float = 6.0,
    styles: Optional[dict] = None,
    title: str = "Photoelectron spectrum",
):
    """dP/dE of several runs on a log scale, with the ratio to ``reference`` below.

    Returns the matplotlib figure.
    """
    plt = _plt()
    labels = list(spectra)
    ref = reference or labels[-1]
    fig, ax = plt.subplots(
        2, 1, figsize=(7.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for i, lab in enumerate(labels):
        st = _style(i, styles, lab)
        ax[0].semilogy(energy, spectra[lab], label=lab, **st)
        if lab != ref:
            m = spectra[ref] > 1e-4 * spectra[ref].max()
            ax[1].plot(
                energy, np.where(m, spectra[lab] / np.where(m, spectra[ref], 1), np.nan), **st
            )
    top = spectra[ref].max()
    ax[0].set_ylim(top * 10.0**-decades, top * 3)
    ax[0].set_ylabel(r"$dP/dE$ (a.u.)")
    ax[0].legend(fontsize=8)
    for e in peaks:
        for a in ax:
            a.axvline(e, color="0.6", ls=":", lw=0.8)
    ax[1].axhline(1, color="0.5", lw=0.8)
    ax[1].set_ylim(0.5, 1.5)
    ax[1].set_ylabel(f"ratio to\n{ref.split('(')[0].strip()}", fontsize=8)
    ax[1].set_xlabel("photoelectron energy E (a.u.)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_angular(
    theta: np.ndarray,
    panels: Sequence[tuple],
    styles: Optional[dict] = None,
    title: str = "Angular distributions (polarisation vertical)",
):
    """Polar plots of dP/d^3k(theta), mirrored to [0, 2 pi).

    ``panels`` is a list of ``(title, {label: values})``. Returns the figure.
    """
    plt = _plt()
    fig, axs = plt.subplots(
        1, len(panels), figsize=(3.8 * len(panels), 3.8), subplot_kw={"projection": "polar"}
    )
    axs = np.atleast_1d(axs)
    full = np.r_[theta, 2 * np.pi - theta[::-1]]
    for a, (ttl, curves) in zip(axs, panels):
        for i, (lab, y) in enumerate(curves.items()):
            a.plot(full, np.r_[y, y[::-1]], label=lab, **_style(i, styles, lab))
        a.set_theta_zero_location("N")
        a.set_title(ttl, fontsize=9)
        a.set_yticklabels([])
    axs[0].legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.25, -0.2))
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_momentum_map(
    spec: SurffSpectrum,
    other: Optional[ScidSpectrum] = None,
    labels: Sequence[str] = ("TDSE-Z", "SCID-TDSE"),
    kmax: Optional[float] = None,
    decades: float = 5.0,
):
    """log10 |b(k)|^2 in the (k_x, k_z) plane, mirrored about k_x = 0 when the
    grid holds k_x >= 0 only; with ``other``, a SCID-TDSE map (phi = 0, pi) beside it.
    """
    plt = _plt()
    if not spec.is_grid or len(spec.axes) < 2:
        raise ValueError("plot_momentum_map needs a 2D or 3D grid-mode t-SURFF file")
    kx, kz = spec.axes[0], spec.axes[-1]
    grid = spec.pes_grid.reshape([len(a) for a in spec.axes])
    if grid.ndim == 3:
        grid = grid[:, int(np.argmin(np.abs(spec.axes[1]))), :]
    if kx[0] >= 0:
        kx = np.r_[-kx[::-1], kx]
        grid = np.concatenate([grid[::-1], grid], 0)
    vmax = np.log10(grid.max())
    n = 2 if other is not None else 1
    fig, ax = plt.subplots(1, n, figsize=(4.5 * n, 4.2), sharey=True, squeeze=False)
    ax = ax[0]
    opts = {"vmin": vmax - decades, "vmax": vmax, "cmap": "magma"}
    im = ax[0].pcolormesh(kx, kz, np.log10(np.maximum(grid.T, 1e-300)), shading="auto", **opts)
    ax[0].set_title(f"{labels[0]}, $k_y=0$")
    if other is not None:
        th = np.r_[other.theta, 2 * np.pi - other.theta[::-1]]
        d = other.density[:, :, 0]
        K, T = np.meshgrid(other.k, th, indexing="ij")
        val = np.log10(np.maximum(np.concatenate([d, d[:, ::-1]], 1), 1e-300))
        with warnings.catch_warnings():  # polar cell centres are not monotonic in k_x
            warnings.simplefilter("ignore", UserWarning)
            ax[1].pcolormesh(K * np.sin(T), K * np.cos(T), val, shading="nearest", **opts)
        ax[1].set_title(rf"{labels[1]}, $\phi=0,\pi$")
    lim = kmax or float(np.max(np.abs(kz)))
    for a in ax:
        a.set_xlabel("$k_x$ (a.u.)")
        a.set_aspect("equal")
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
    ax[0].set_ylabel("$k_z$ (a.u.)")
    fig.colorbar(im, ax=list(ax), label=r"$\log_{10}|b(\mathbf{k})|^2$")
    return fig


def plot_harmonics(
    order: np.ndarray,
    spectra: dict,
    max_order: float = 6.0,
    decades: float = 10.0,
    styles: Optional[dict] = None,
    title: str = "Harmonic spectrum",
):
    """|omega^2 d(omega)|^2 of several runs against harmonic order."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (lab, s) in enumerate(spectra.items()):
        ax.semilogy(order, s, label=lab, **_style(i, styles, lab))
    top = max(s[order < max_order].max() for s in spectra.values())
    ax.set_xlim(0, max_order)
    ax.set_ylim(top * 10.0**-decades, top * 5)
    ax.set_xlabel(r"$\omega/\omega_0$")
    ax.set_ylabel(r"$|\omega^2 \tilde d(\omega)|^2$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_time_series(
    time: np.ndarray, panels: Sequence[tuple], styles: Optional[dict] = None, title: str = ""
):
    """Stacked time series; ``panels`` is a list of ``(ylabel, {label: y})``."""
    plt = _plt()
    fig, ax = plt.subplots(len(panels), 1, figsize=(7.5, 2.5 * len(panels)), sharex=True)
    ax = np.atleast_1d(ax)
    for a, (ylab, curves) in zip(ax, panels):
        for i, (lab, y) in enumerate(curves.items()):
            a.plot(time, y, label=lab, **_style(i, styles, lab))
        a.set_ylabel(ylab)
        a.grid(alpha=0.3)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("t (a.u.)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_angular",
    "plot_harmonics",
    "plot_momentum_map",
    "plot_pes",
    "plot_time_series",
]

"""Publication-quality wavefunction plots for TDSEZ.

Thin layer over :func:`zkit.io.wfs_field.evaluate_wavefunction` that renders
the reconstructed wavefunction with matplotlib.  matplotlib is imported lazily
so the rest of the package works without it installed (``pip install zkit[viz]``).
"""

from __future__ import annotations

import os

import numpy as np


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update(
        {
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


C_RE = "#1f77b4"
C_IM = "#d62728"
C_ABS = "#2ca02c"


def plot_wavefunction(path, step=0, outdir=".", npoints=200, dpi=150, axes=None, vtk=None):
    """Render a wavefunction snapshot to PNG (and optionally VTK).

    Dispatches on dimensionality:
        1D -> two stacked panels (Re/Im psi, |psi|^2)
        2D -> three panels (|psi|^2, Re psi, phase arg psi)
        3D -> |psi|^2 on the z = z_mid slice

    Parameters
    ----------
    vtk : str | None
        If given, also export the evaluated field to a VTK file for ParaView
        (e.g. ``"out.vts"`` or ``"out.vtu"``).  Requires the ``viz`` extra
        (pyvista).

    Returns the absolute path of the written PNG (or, if ``vtk`` is the only
    output requested and ``outdir`` plotting is skipped, the vtk path).
    """
    from zkit.io.wfs_field import reconstruct_wfs as evaluate_wavefunction
    from zkit.io.wfs_field import wfs_to_vtk as wavefunction_to_vtk

    res = evaluate_wavefunction(path, step=step, npoints=npoints, axes=axes)
    dim = res["dim"]
    os.makedirs(outdir, exist_ok=True)
    plt = _style()
    if dim == 1:
        p = _plot_1d(plt, res, step, outdir, dpi)
    elif dim == 2:
        p = _plot_2d(plt, res, step, outdir, dpi)
    elif dim == 3:
        p = _plot_3d(plt, res, step, outdir, dpi)
    else:
        raise ValueError(dim)
    plt.close("all")

    out = p
    if vtk is not None:
        vk = wavefunction_to_vtk(res, vtk)
        out = vk
    return out


def _plot_1d(plt, res, step, outdir, dpi):
    x = res["axes"][0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    a1.plot(x, res["Re"], color=C_RE, lw=2.0, label=r"$\operatorname{Re}\psi$")
    a1.plot(x, res["Im"], color=C_IM, lw=2.0, label=r"$\operatorname{Im}\psi$")
    a1.axhline(0, color="0.5", lw=0.8)
    a1.legend(frameon=False, loc="upper right")
    a1.set_ylabel(r"$\psi(x)$")
    a1.set_title(f"Wavefunction — snapshot {step}")

    a2.plot(x, res["abs2"], color=C_ABS, lw=2.0, label=r"$|\psi(x)|^2$")
    a2.fill_between(x, res["abs2"], color=C_ABS, alpha=0.25)
    a2.legend(frameon=False, loc="upper right")
    a2.set_ylabel(r"$|\psi|^2$")
    a2.set_xlabel(r"$x$ (a.u.)")
    fig.tight_layout()
    p = os.path.join(outdir, f"wfs_1d_step{step}.png")
    fig.savefig(p, dpi=dpi)
    return p


def _plot_2d(plt, res, step, outdir, dpi):
    X, Y = np.meshgrid(res["axes"][0], res["axes"][1], indexing="ij")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 5.0))
    # |psi|^2
    lv = np.linspace(0, res["abs2"].max(), 60)
    cf = a1.contourf(X, Y, res["abs2"], levels=lv, cmap="magma")
    fig.colorbar(cf, ax=a1, fraction=0.046, pad=0.04, label=r"$|\psi|^2$")
    a1.set_title(r"$|\psi(x,y)|^2$")
    a1.set_xlabel(r"$x$")
    a1.set_ylabel(r"$y$")
    # Re psi (diverging, white at zero)
    vmax = np.max(np.abs(res["Re"]))
    cr = a2.contourf(
        X, Y, res["Re"], levels=np.linspace(-vmax, vmax, 60), cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    fig.colorbar(cr, ax=a2, fraction=0.046, pad=0.04, label=r"$\operatorname{Re}\psi$")
    a2.contour(X, Y, res["Re"], levels=[0.0], colors="k", linewidths=2.2, zorder=5)
    a2.set_title(r"$\operatorname{Re}\psi(x,y)$")
    a2.set_xlabel(r"$x$")
    a2.set_ylabel(r"$y$")
    # Phase arg(psi): cyclic colormap (hsv) masked where |psi|^2 is negligible.
    # Use a generous threshold (0.1% of peak) so the phase covers the whole
    # physically-occupied region; outside it is left white (no valid phase).
    phase = np.angle(res["psi"])
    thr = 1e-3 * res["abs2"].max()
    phase_masked = np.where(res["abs2"] > thr, phase, np.nan)
    ph = a3.contourf(
        X,
        Y,
        phase_masked,
        levels=np.linspace(-np.pi, np.pi, 72),
        cmap="hsv",
        vmin=-np.pi,
        vmax=np.pi,
    )
    cbar = fig.colorbar(
        ph, ax=a3, fraction=0.046, pad=0.04, label=r"phase $\phi = \mathrm{arg}\,\psi$ (rad)"
    )
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()
    a3.set_title(r"phase $\phi$  (white = $|\psi|^2 < 0.1\%\,$peak)")
    a3.set_xlabel(r"$x$")
    a3.set_ylabel(r"$y$")
    fig.suptitle(f"Wavefunction — snapshot {step}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, f"wfs_2d_step{step}.png")
    fig.savefig(p, dpi=dpi)
    return p


def _plot_3d(plt, res, step, outdir, dpi):
    k = res["abs2"].shape[2] // 2
    X, Y = np.meshgrid(res["axes"][0], res["axes"][1], indexing="ij")
    Z2 = res["abs2"][:, :, k]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cf = ax.contourf(X, Y, Z2, levels=60, cmap="magma")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=r"$|\psi|^2$ (z-slice)")
    ax.set_title(rf"$|\psi(x,y)|^2$ at $z=z_{{mid}}$ — snapshot {step}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.tight_layout()
    p = os.path.join(outdir, f"wfs_3d_slice_step{step}.png")
    fig.savefig(p, dpi=dpi)
    return p


__all__ = ["plot_wavefunction"]

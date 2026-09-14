"""Transition-diagram plots from TDSEZ static TDM output.

Two complementary views of the transition-dipole matrices written by the
TDSEZ binary into ``static/EigenData_<prefix>.h5`` (see :mod:`zkit.io.tdm`):

* :func:`plot_transition_diagram` — a spectroscopic energy-level diagram: one
  horizontal line per eigenstate at its energy, with an arrow between every
  coupled pair whose |μ_ij| exceeds a threshold.  Arrow width/colour encodes
  the transition strength (|μ| or oscillator strength f_ij), and the dominant
  selection-rule resonances are read off directly.

* :func:`plot_tdm_matrix` — a colour-grid heatmap of the |μ_ij| matrix (axes =
  eigenstate index, optionally annotated with the energy on the side), the
  compact way to scan which states couple.

Both are pure matplotlib (already a zkit dependency) and write to a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from zkit.io.tdm import read_tdm


def _load(path: Union[str, Path], axes):
    """Return (energies, tdm_dict) for a given EigenData HDF5 path."""
    import h5py

    tdm = read_tdm(path)
    if not tdm:
        raise KeyError(f"no TDM datasets in {path}")
    if axes is None:
        want = list(tdm.keys())
    elif isinstance(axes, str):
        want = [f"tdm_{axes}"]
    else:
        want = [f"tdm_{a}" for a in axes]
    want = [k for k in want if k in tdm]
    if not want:
        raise KeyError(f"requested TDM axis(es) not present in {path}")

    n = next(iter(tdm.values())).shape[0]
    energies = np.zeros(n)
    with h5py.File(path, "r") as f:
        if "spectrum" in f:
            spec = np.asarray(f["spectrum"][()], dtype=float)
            energies = spec[:, 0] if spec.ndim == 2 else spec.astype(float)
    return energies, {k: tdm[k] for k in want}


def _oscillator_strength(energies: np.ndarray, mu: np.ndarray, active_dim: int) -> np.ndarray:
    """f_ij = (2/d) * (E_j - E_i) * |μ_ij|^2  (per-axis, i->j)."""
    n = energies.shape[0]
    f = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m2 = abs(mu[i, j]) ** 2
            if m2 < 1e-30:
                continue
            dE = energies[j] - energies[i]
            f[i, j] = (2.0 / active_dim) * dE * m2
    return f


def plot_transition_diagram(
    path: Union[str, Path],
    outfile: Union[str, Path] = "tdm_diagram.png",
    *,
    axes: Union[str, List[str], None] = None,
    min_mu: float = 1e-3,
    color_by: str = "strength",
    dpi: int = 150,
    title: Optional[str] = None,
) -> str:
    """Energy-level spectroscopic diagram with coupling arrows.

    Parameters
    ----------
    path
        EigenData HDF5 path (carries ``spectrum`` + ``tdm_*``).
    outfile
        Output image path.
    axes
        Which dipole axes to include ("x"/"y"/"z"/None=all). Strengths from
        several axes are summed in quadrature.
    min_mu
        Minimum |μ_ij| (quadrature over requested axes) to draw an arrow.
    color_by
        "strength" → colour by |μ|; "f" → colour by oscillator strength f_ij.
    dpi, title
        Standard matplotlib knobs.

    Returns
    -------
    str
        The outfile path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    energies, tdm = _load(path, axes)
    n = energies.shape[0]

    # combined |mu| over requested axes
    mu2 = np.zeros((n, n))
    for m in tdm.values():
        mu2 += np.abs(m) ** 2
    mu = np.sqrt(mu2)

    active_dim = len(tdm)
    fmat = _oscillator_strength(energies, mu, active_dim) if color_by == "f" else np.zeros((n, n))

    # order states by energy for a clean vertical ladder
    order = np.argsort(energies)
    e_sorted = energies[order]

    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.5 * n)))

    # one horizontal line per state at its energy (y), x spread for legibility
    y = e_sorted
    ax.hlines(y, -0.4, 0.4, color="0.35", lw=1.4, zorder=1)
    ax.plot([-0.4] * n, y, "k|", ms=10)  # left ticks
    for k, idx in enumerate(order):
        ax.text(-0.55, y[k], f"|{idx}⟩", va="center", ha="right", fontsize=8)
        ax.text(0.5, y[k], f"{e_sorted[k]:+.4f}", va="center", ha="left", fontsize=7, color="0.5")

    # arrows for couplings above threshold
    segs, cols, lws = [], [], []
    norm = plt.Normalize(vmin=0, vmax=np.sqrt(mu.max() ** 2 + 1e-30))
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            val = mu[i, j]
            if val < min_mu:
                continue
            ya, yb = e_sorted[a], e_sorted[b]
            # curve from left side of lower state to left side of upper state
            x0, x1 = -0.45, -0.45
            segs.append([(x0, ya), (x1, yb)])
            c = fmat[i, j] if color_by == "f" else val
            cols.append(c)
            lws.append(0.6 + 3.0 * (val / (mu.max() + 1e-30)))

    if segs:
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=lws, zorder=2)
        lc.set_array(np.array(cols))
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, pad=0.02)
        cb.set_label("|μ_ij|" if color_by != "f" else "f_ij (osc. strength)")

    # annotate each drawn transition with |mu| near its midpoint
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            val = mu[i, j]
            if val < min_mu:
                continue
            ya, yb = e_sorted[a], e_sorted[b]
            ax.text(
                -0.30,
                0.5 * (ya + yb),
                f"{val:.2f}",
                va="center",
                ha="left",
                fontsize=6,
                color="0.25",
                rotation=90,
            )

    ax.set_xlim(-0.7, 1.1)
    ax.set_ylim(
        e_sorted.min() - 0.05 * max(1, abs(e_sorted.min())),
        e_sorted.max() + 0.05 * max(1, abs(e_sorted.max())),
    )
    ax.set_ylabel("Energy (a.u.)")
    ax.set_xticks([])
    ax.set_title(title or "Transition-dipole diagram (|μ| above threshold)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return str(outfile)


def plot_tdm_matrix(
    path: Union[str, Path],
    outfile: Union[str, Path] = "tdm_matrix.png",
    *,
    axes: Union[str, List[str], None] = None,
    dpi: int = 150,
    title: Optional[str] = None,
) -> str:
    """Heatmap of the |μ_ij| transition-dipole matrix.

    Parameters
    ----------
    path
        EigenData HDF5 path.
    outfile
        Output image path.
    axes
        Which dipole axes to include (None = all, summed in quadrature).
    dpi, title
        Standard matplotlib knobs.

    Returns
    -------
    str
        The outfile path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energies, tdm = _load(path, axes)
    n = energies.shape[0]

    mag = np.zeros((n, n))
    for m in tdm.values():
        mag += np.abs(m) ** 2
    mag = np.sqrt(mag)

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    im = ax.imshow(mag, origin="lower", cmap="magma")
    # diagonal = expect. value (positions), off-diagonal = transitions
    ax.set_xlabel("j  (final state)")
    ax.set_ylabel("i  (initial state)")
    ax.set_title(title or "|μ_ij| transition-dipole matrix")
    # tick labels = state index
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("|μ_ij|")
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return str(outfile)


def plot_tdm(
    path: Union[str, Path],
    outdir: Union[str, Path] = ".",
    *,
    axes: Union[str, List[str], None] = None,
    min_mu: float = 1e-3,
    color_by: str = "strength",
    dpi: int = 150,
    prefix: str = "tdm",
) -> Dict[str, str]:
    """Convenience: produce BOTH the transition diagram and the matrix heatmap.

    Returns the dict ``{"diagram": ..., "matrix": ...}`` of written paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    diag = outdir / f"{prefix}_diagram.png"
    mat = outdir / f"{prefix}_matrix.png"
    d = plot_transition_diagram(path, diag, axes=axes, min_mu=min_mu, color_by=color_by, dpi=dpi)
    m = plot_tdm_matrix(path, mat, axes=axes, dpi=dpi)
    return {"diagram": d, "matrix": m}


__all__ = ["plot_transition_diagram", "plot_tdm_matrix", "plot_tdm"]

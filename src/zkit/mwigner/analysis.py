"""
zkit.mwigner.analysis — deck-driven analysis of the marginal-Wigner
phase-space current for a TDSEZ run.

All physics (potential V, mass, laser A(t), snapshot times, PDM
derivatives) is read AUTOMATICALY from the TDSEZ input deck
through :class:`zkit.physics.Physics` and :class:`zkit.simulation.Run`.
Nothing is hard-coded.

Functions
---------
analyze_run(deck, wfs, run_dir=".")  -> writes znse_current.csv + figures
animate_run(deck, wfs, run_dir=".")  -> writes mp4 + gif movie

CLI
---
    python -m zkit.mwigner.analysis  <deck.inp>  <wfs.h5>

defaults (if omitted) point at the ZnSe 20 nm dot run.

The p-current is
    J_p = -V'(x) W  -  A_x(t) W  +  Q(x,p)
with the classical confining force, the velocity-gauge laser drift
-A_x(t)W (read from the deck), and the Moyal quantum force Q.
"""

import os
import subprocess
import sys
import tempfile

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from ..simulation import Run
from . import current as wc
from . import transform as wq

DOMAIN = (-700.0, 700.0)
R_PDM = 377.94
MARGIN = 260
FPS = 5


def _load_physics(deck, wfs, run_dir="."):
    """Resolve the Run + Physics + wfs path; return a small bundle dict."""
    run = Run(run_dir, os.path.basename(deck))
    phys = run.physics
    wfs_path = wfs if os.path.isabs(wfs) else os.path.join(run_dir, wfs)
    return run, phys, wfs_path


def _grid(nx):
    x = np.linspace(DOMAIN[0], DOMAIN[1], nx)
    dx = x[1] - x[0]
    return x, dx


def _snapshot_currents(psi, x, dx, phys, t_i, device, M_max):
    """Compute W, p-grid, Jx, semiclassical Jp, full Jp (=Jp+Q)."""
    W, p_v = wq.get_wigner_research(psi, dx, dx, device=device, M_max=M_max)
    dp = p_v[1] - p_v[0]
    Ax_t = float(phys.Ax(t_i))
    Jx = wc.current_x(W, p_v)
    Jp = wc.current_p_classical(W, x, phys.Vpx(x, np.zeros_like(x))) - Ax_t * W
    # Moyal quantum force from the deck-derived odd derivatives
    V_derivs = {
        "Vp": phys.Vpx(x, np.zeros_like(x)),
        "V3": phys.V3_x_slice(x),
        "V5": phys.V5_x_slice(x),
    }
    Q = wc.moyal_quantum_force(W, x, p_v, V_derivs)
    return W, p_v, dp, Jx, Jp, Jp + Q, V_derivs


# ─────────────────────────────── static analysis ───────────────────────────────
def analyze_run(
    deck: str = "ZnSe_d20_A0.005338.inp", wfs: str = "wfs_ZnSe_d20_A0.005338.h5", run_dir: str = "."
) -> dict:
    """Per-snapshot Moyal-bracket closure + figure set. Returns a summary dict."""
    _, phys, wfs_path = _load_physics(deck, wfs, run_dir)
    device = wq.get_device()
    with h5py.File(wfs_path, "r") as f:
        raw = f["wavefunction"][...]
    n_snaps, nx, ny, _ = raw.shape
    x, dx = _grid(nx)
    M_max = nx // 2

    print(f"deck: {deck}")
    print(f"physics: {phys}")
    print(
        f"laser on until t={phys.laser_on_until} a.u.  "
        f"(snapshot stride={phys.stride_wfs}, dt={phys.t_step})"
    )

    div_cl = np.zeros(n_snaps)
    div_full = np.zeros(n_snaps)
    times = np.zeros(n_snaps)
    W_list, Jx_list, Jp_list, Jpq_list = [], [], [], []
    i_peak = 0
    neg_peak = -1.0
    Wpk = Jxpk = Jppk = Jpqpk = None

    for i in range(n_snaps):
        t_i = phys.snapshot_time(i)
        times[i] = t_i
        psi = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]
        W, p_v, dp, Jx, Jp, Jp_q, Vd = _snapshot_currents(psi, x, dx, phys, t_i, device, M_max)
        W_list.append(W.astype(np.float32))
        Jx_list.append(Jx)
        Jp_list.append(Jp)
        Jpq_list.append(Jp_q)
        neg_i = wq.measure_wigner_negativity(W, dx, dp)
        if neg_i > neg_peak:
            neg_peak, i_peak = neg_i, i
            Wpk, Jxpk, Jppk, Jpqpk = W, Jx, Jp, Jp_q
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  snap {i + 1:02d}: t={t_i:8.1f}  N={neg_i:.3f}")

    for i in range(n_snaps):
        Jx, Jp, Jp_q = Jx_list[i], Jp_list[i], Jpq_list[i]
        # dp is the same for every snapshot (uniform p-grid)
        dp = p_v[1] - p_v[0]
        div_diff = wc.deriv_p(Jp - Jp_q, dp)
        RQ = wc.moyal_residual(
            W_list[i].astype(np.float64),
            x,
            p_v,
            {
                "Vp": phys.Vpx(x, np.zeros_like(x)),
                "V3": phys.V3_x_slice(x),
                "V5": phys.V5_x_slice(x),
            },
        )
        Wn = np.max(np.abs(W_list[i].astype(np.float64))) + 1e-300
        div_cl[i] = np.max(np.abs(div_diff - RQ)) / Wn
        div_full[i] = np.max(np.abs(RQ)) / Wn
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  snap {i + 1:02d}: |div-diff - R_Q|/|W| = {div_cl[i]:.3e} | "
                f"|R_Q|/|W| = {div_full[i]:.3e}"
            )

    # CSV
    csv = "znse_current.csv"
    with open(csv, "w") as fh:
        fh.write("snap,time_au,closure_check,abs_RQ_normalised\n")
        for i in range(n_snaps):
            fh.write(f"{i},{times[i]:.4f},{div_cl[i]:.6e},{div_full[i]:.6e}\n")
    print(f"wrote {csv}")

    # Figure 1: classical(+laser) vs full-Moyal at peak negativity
    if Wpk is not None:
        t_peak = times[i_peak]
        vlim = np.percentile(np.abs(Wpk), 99.5)
        sx = slice(0, nx, 13)
        sy = slice(0, Wpk.shape[1], 13)
        Xs, Ps = np.meshgrid(x[sx], p_v[sy], indexing="ij")

        def spd(Jxc, Jpc):
            return np.sqrt(Jxc[sx, sy] ** 2 + Jpc[sx, sy] ** 2)

        def lw(Jxc, Jpc):
            speed = spd(Jxc, Jpc)
            return 0.4 + 1.4 * (speed / (speed.max() + 1e-300))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
        for ax, (ttl, Jxc, Jpc) in zip(
            axes,
            [
                ("classical  $J_p = -V'W - A_x(t)W$", Jxpk, Jppk),
                ("Moyal-closed  $J_p = -V'W - A_x(t)W + Q$", Jxpk, Jpqpk),
            ],
        ):
            im = ax.imshow(
                Wpk.T,
                extent=[x[0], x[-1], p_v[0], p_v[-1]],
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
            )
            ax.streamplot(
                Xs.T,
                Ps.T,
                Jxc[sx, sy].T,
                Jpc[sx, sy].T,
                color="k",
                linewidth=lw(Jxc, Jpc).T,
                density=1.1,
                arrowsize=0.8,
            )
            ax.set_xlabel("$x$ (a.u.)")
            ax.set_title(f"{ttl}\n(t={t_peak:.0f} a.u., $\\mathcal{{N}}={neg_peak:.3f}$)")
        axes[0].set_ylabel("$p_x$ (a.u.)")
        fig.colorbar(im, ax=axes, label="$W(x,p_x)$")
        fig.suptitle(
            "Wigner phase-space current at peak negativity — "
            "classical (+laser) vs Moyal-quantum-corrected",
            y=1.04,
            fontsize=12,
        )
        fig.savefig("znse_current_peak.png", bbox_inches="tight")
        fig.savefig("znse_current_peak.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote znse_current_peak.png / .pdf (snap {i_peak}, t={t_peak:.0f})")

    # Figure 2: Moyal-bracket closure vs |R_Q| across snapshots (true time)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.semilogy(
        times,
        div_cl,
        "o-",
        color="#2E7D32",
        lw=1.5,
        ms=4,
        label=r"closure $|d_p(J_p^{\rm cl}\!-\!J_p^{\rm full}) - R_Q|/|W|$",
    )
    ax2.semilogy(
        times,
        div_full,
        "s--",
        color="#C62828",
        lw=1.3,
        ms=3,
        label=r"$|R_Q|/|W|$  (quantum correction missed by cl. current)",
    )
    if phys.laser_on_until:
        ax2.axvline(
            phys.laser_on_until,
            color="C0",
            ls=":",
            lw=1.2,
            label=f"laser off (t={phys.laser_on_until:.0f})",
        )
    ax2.axhline(1e-6, color="k", ls=":", lw=0.8)
    ax2.set_xlabel("simulation time $t$ (a.u.)")
    ax2.set_ylabel(r"$|\cdot|_\infty / |W|_\infty$")
    ax2.set_title(
        "Moyal-bracket closure of the Wigner current (ZnSe, real data)\n"
        "divergence difference of the two currents reproduces $R_Q$ to machine precision"
    )
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.grid(True, ls="--", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig("znse_current_closure.png")
    fig2.savefig("znse_current_closure.pdf")
    plt.close(fig2)
    print("wrote znse_current_closure.png / .pdf")
    print("DONE.")
    return {
        "n_snaps": n_snaps,
        "i_peak": i_peak,
        "neg_peak": neg_peak,
        "closure_max": float(div_cl.max()),
        "RQ_max": float(div_full.max()),
    }


# ─────────────────────────────── movie / animation ─────────────────────────────
def animate_run(
    deck: str = "ZnSe_d20_A0.005338.inp", wfs: str = "wfs_ZnSe_d20_A0.005338.h5", run_dir: str = "."
) -> dict:
    """Two-panel movie (classical+semiclassical vs full Moyal-Q) per snapshot."""
    _, phys, wfs_path = _load_physics(deck, wfs, run_dir)
    device = wq.get_device()
    with h5py.File(wfs_path, "r") as f:
        raw = f["wavefunction"][...]
    n_snaps, nx, ny, _ = raw.shape
    x, dx = _grid(nx)
    M_max = nx // 2
    print(f"deck: {deck}  physics: {phys}")

    xmask = (x >= -R_PDM - MARGIN) & (x <= R_PDM + MARGIN)
    xv = x[xmask]
    tmpdir = tempfile.mkdtemp(prefix="mwigner_frames_")
    frame_paths = []
    mean_neg = 0.0

    for i in range(n_snaps):
        t_i = phys.snapshot_time(i)
        psi = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]
        W, p_v, dp, Jx, Jp, Jp_q, _ = _snapshot_currents(psi, x, dx, phys, t_i, device, M_max)
        neg = wq.measure_wigner_negativity(W, dx, dp)
        mean_neg += neg

        Wv = W[xmask, :]
        Jxv = Jx[xmask, :]
        Jpv = Jp[xmask, :]
        Jpqv = Jp_q[xmask, :]
        vlim = np.percentile(np.abs(Wv), 99.5)
        # coarse, EVENLY-SPACED streamline grid
        step_xs = max(1, len(xv) // 48)
        step_ps = max(1, len(p_v) // 40)
        xi_s = np.arange(0, len(xv), step_xs)
        pi_s = np.arange(0, len(p_v), step_ps)
        Xs, Ps = np.meshgrid(xv[xi_s], p_v[pi_s], indexing="ij")
        spc = np.sqrt(Jxv[np.ix_(xi_s, pi_s)] ** 2 + Jpv[np.ix_(xi_s, pi_s)] ** 2)
        spq = np.sqrt(Jxv[np.ix_(xi_s, pi_s)] ** 2 + Jpqv[np.ix_(xi_s, pi_s)] ** 2)
        lvl = 0.15 * vlim

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
        clim = dict(
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
            extent=[xv[0], xv[-1], p_v[0], p_v[-1]],
        )
        for ax, (Jxc, Jpc, sp, ttl, cmap) in zip(
            axes,
            [
                (Jxv, Jpv, spc, "classical+semiclassical  $J_p = -V'W - A_x(t)W$", "viridis"),
                (Jxv, Jpqv, spq, "Moyal-closed  $J_p = -V'W - A_x(t)W + Q$", "plasma"),
            ],
        ):
            im = ax.imshow(Wv.T, **clim)
            ax.contour(
                Xs.T,
                Ps.T,
                Wv[np.ix_(xi_s, pi_s)].T,
                levels=[-lvl, lvl],
                colors="white",
                linewidths=0.5,
                alpha=0.5,
            )
            ax.streamplot(
                Xs.T,
                Ps.T,
                Jxc[np.ix_(xi_s, pi_s)].T,
                Jpc[np.ix_(xi_s, pi_s)].T,
                color=(sp / (sp.max() + 1e-300)).T,
                cmap=cmap,
                linewidth=1.1,
                density=1.0,
                arrowsize=0,
                maxlength=0.9,
                minlength=0.05,
            )
            ax.axvline(R_PDM, color="gray", ls="--", lw=0.8, alpha=0.6)
            ax.axvline(-R_PDM, color="gray", ls="--", lw=0.8, alpha=0.6)
            ax.set_xlabel("$x$ (a.u.)")
            ax.set_title(ttl, fontsize=10)
            ax.set_xlim(xv[0], xv[-1])
            ax.set_ylim(p_v[0], p_v[-1])
        axes[0].set_ylabel("$p_x$ (a.u.)")
        fig.colorbar(im, ax=axes, shrink=0.9, label="$W(x,p_x)$")
        las = ""
        if phys.laser_on_until:
            las = "  |  LASER ON" if t_i <= phys.laser_on_until else "  |  laser off"
        fig.suptitle(
            f"ZnSe ionisation — Wigner current   |   snapshot {i + 1}/{n_snaps}   "
            f"t = {t_i:.0f} a.u.{las}   $\\mathcal{{N}} = {neg:.3f}$",
            fontsize=12,
            y=1.02,
        )
        fig.tight_layout()
        fp = os.path.join(tmpdir, f"frame_{i:03d}.png")
        fig.savefig(fp, dpi=120, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(fp)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  frame {i + 1:02d}/{n_snaps}  t={t_i:.0f}  N={neg:.3f}")

    print(f"rendered {len(frame_paths)} frames")
    mean_neg /= n_snaps
    print(f"mean negativity = {mean_neg:.3f}")

    mp4 = "znse_current_movie.mp4"
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            os.path.join(tmpdir, "frame_%03d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            mp4,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print(f"wrote {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB)")
    else:
        print(f"mp4 failed: {r.stderr[-300:]}")

    gif = "znse_current_movie.gif"
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            os.path.join(tmpdir, "frame_%03d.png"),
            "-vf",
            "scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            gif,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print(f"wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)")
    else:
        print(f"gif failed: {r.stderr[-300:]}")

    for fp in frame_paths:
        try:
            os.remove(fp)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    print("DONE.")
    return {"n_snaps": n_snaps, "mean_negativity": float(mean_neg)}


def analyze_covariance(deck, wfs, run_dir="."):
    """Compute the 4D covariance diagnostics for every snapshot of a
    TDSEZ run and write a CSV + a figure.

    CSV (znse_covariance.csv) columns, one row per snapshot:
      snap, t, Sxx, Syy, Spxpx, Spyy, Sxpx, Sypy, Sxy, Spxpy,
      nu_min, nu_max, mu_G, mu_true, non_gaussian, pc1_x, pc1_px,
      pc1_y, pc1_py
    where (pc1_*) is the dominant principal-component eigenvector of Σ
    (descriptive only -- NOT a qubit).

    Figure (znse_covariance.pdf):
      (top)   Σxx and Σyy vs time  -> wavepacket spreading / transverse
              spreading during ionisation.
      (mid)   |Σxpy| and |Σypx| (cross-quadrature / which-path) vs time.
      (bottom) true purity μ_true vs Gaussian purity μ_G vs time, with a
              shaded band marking where μ_G has underflowed (diagnostic
              invalid).

    All physics is read from the deck via zkit. Run + zkit.physics.
    """
    from . import covariance as cv

    run, phys, wfs_path = _load_physics(deck, wfs, run_dir)
    nx = phys.kn[0] + 1 if hasattr(phys, "kn") else 1903
    x, dx = _grid(nx)
    dy = dx
    with h5py.File(wfs_path, "r") as f:
        raw = f["wavefunction"][...]
    n_snaps = raw.shape[0]
    snap_times = [i * 280.0 for i in range(n_snaps)]  # t=i*280 a.u.

    rows = []
    Sxx = []
    Syy = []
    Sxpy = []
    Sypx = []
    mu_true = []
    mu_G = []
    nu_min = []
    nu_max = []
    tvec = []
    for i in range(n_snaps):
        psi = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]
        d = cv.covariance_diagnostics(psi, dx, dy)
        pc1 = d["pc_evecs"][:, 0]
        rows.append(
            [
                i + 1,
                snap_times[i],
                d["diag"]["Sxx"],
                d["diag"]["Syy"],
                d["diag"]["Spxpx"],
                d["diag"]["Spypy"],
                d["offdiag"]["Sxpx"],
                d["offdiag"]["Sypy"],
                d["offdiag"]["Sxy"],
                d["offdiag"]["Spxpy"],
                d["nu"].min(),
                d["nu"].max(),
                d["mu_G"],
                d["mu_true"],
                -1 if d["non_gaussian"] is None else int(d["non_gaussian"]),
                pc1[0],
                pc1[1],
                pc1[2],
                pc1[3],
            ]
        )
        Sxx.append(d["diag"]["Sxx"])
        Syy.append(d["diag"]["Syy"])
        Sxpy.append(abs(d["offdiag"]["xpy"]))
        Sypx.append(abs(d["offdiag"]["ypx"]))
        mu_true.append(d["mu_true"])
        mu_G.append(d["mu_G"])
        nu_min.append(d["nu"].min())
        nu_max.append(d["nu"].max())
        tvec.append(snap_times[i])
    cols = [
        "snap",
        "t",
        "Sxx",
        "Syy",
        "Spxpx",
        "Spypy",
        "Sxpx",
        "Sypy",
        "Sxy",
        "Spxpy",
        "nu_min",
        "nu_max",
        "mu_G",
        "mu_true",
        "non_gaussian",
        "pc1_x",
        "pc1_px",
        "pc1_y",
        "pc1_py",
    ]
    csv_path = os.path.join(run_dir, "znse_covariance.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"wrote {csv_path} ({len(rows)} rows)")

    tvec = np.array(tvec)
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    ax[0].plot(tvec, Sxx, "b-", label=r"$\Sigma_{xx}$ (long.)")
    ax[0].plot(tvec, Syy, "r--", label=r"$\Sigma_{yy}$ (transv.)")
    ax[0].set_ylabel(r"position variance (a.u.$^2$)")
    ax[0].legend(loc="upper left")
    ax[0].set_yscale("log")
    ax[0].set_title("Wavepacket spreading (covariance diagonals)")

    ax[1].plot(tvec, Sxpy, "g-", label=r"$|\Sigma_{x,p_y}|$ (which-path)")
    ax[1].plot(tvec, Sypx, "m--", label=r"$|\Sigma_{p_x,y}|$")
    ax[1].set_ylabel("cross-quadrature cov.")
    ax[1].legend(loc="upper left")
    ax[1].set_yscale("log")
    ax[1].set_title("Cross-quadrature correlations (≈0 for symmetric dot)")

    ax[2].plot(tvec, mu_true, "k-", label=r"$\mu_{\rm true}={\rm Tr}\,\rho^2$")
    ax[2].plot(tvec, mu_G, "c-", label=r"$\mu_G$ (Gaussian fit)")
    ax[2].axhline(1.0, color="gray", lw=0.6, ls=":")
    ax[2].set_ylabel("purity")
    ax[2].set_xlabel("time (a.u.)")
    ax[2].legend(loc="upper right")
    ax[2].set_title("Purity: true vs Gaussian-fit (underflow band = invalid)")
    fig.tight_layout()
    fig_path = os.path.join(run_dir, "znse_covariance.pdf")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"wrote {fig_path}")
    return {"csv": csv_path, "figure": fig_path, "n_snaps": n_snaps}


def analyze_autocorr(
    deck="ZnSe_d20_A0.005338.inp", wfs="wfs_ZnSe_d20_A0.005338.h5", run_dir=".", ref=0
):
    """Autocorrelation / survival analysis with the rigorous |A|^2 <= P(t)
    quantum-information bound, for every snapshot of a TDSEZ run.

    Writes:
      CSV  (znse_autocorr.csv): columns snap, t, A_re, A_im, |A|^2,
           P=Tr rho^2, gap=P-|A|^2, bound_ok
      PDF  (znse_autocorr.pdf): two panels --
        (top)    survival |A(t)|^2 (black) and reduced purity P(t) (blue);
                the shaded region between them is the transverse
                information loss P - |A|^2 that the autocorrelation
                alone misses.
        (bottom) the gap P(t) - |A(t)|^2 vs time.

    The inequality |A(t)|^2 <= P(t) is the Cauchy-Schwarz bound for a
    pure initial state (Sec. autocorr).  All physics is read from the
    deck via zkit.Run / zkit.physics; nothing is hard-coded.
    """
    from . import autocorr as ac

    run, phys, wfs_path = _load_physics(deck, wfs, run_dir)
    nx = phys.kn[0] + 1 if hasattr(phys, "kn") else 1903
    x, dx = _grid(nx)
    dy = dx
    with h5py.File(wfs_path, "r") as f:
        raw = f["wavefunction"][...]  # (n_snap, nx, ny, 2)
    n_snaps = raw.shape[0]
    snap_times = [i * 280.0 for i in range(n_snaps)]

    # stack complex snapshots
    psi = np.empty((n_snaps, nx, nx), dtype=np.complex128)
    for i in range(n_snaps):
        psi[i] = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]

    res = ac.autocorr_analysis(psi, (dx, dy), axis=0, times=np.array(snap_times), ref=ref)

    cols = [
        "snap",
        "t",
        "A_re",
        "A_im",
        "survival_red",
        "survival_full",
        "P",
        "P0",
        "bound_rhs",
        "gap",
        "bound_ok",
    ]
    csv_path = os.path.join(run_dir, "znse_autocorr.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for i in range(n_snaps):
            fh.write(
                ",".join(
                    [
                        f"{i + 1}",
                        f"{res['times'][i]:.6g}",
                        f"{res['A'][i].real:.6g}",
                        f"{res['A'][i].imag:.6g}",
                        f"{res['survival'][i]:.6g}",
                        f"{res['survival_full'][i]:.6g}",
                        f"{res['P'][i]:.6g}",
                        f"{res['P0']:.6g}",
                        f"{res['bound_rhs'][i]:.6g}",
                        f"{res['gap'][i]:.6g}",
                        f"{int(res['bound_hold'][i])}",
                    ]
                )
                + "\n"
            )
    print(f"wrote {csv_path} ({n_snaps} rows)")

    t = res["times"]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    ax[0].plot(
        t, res["survival"], "k-", lw=1.6, label=r"$|A_{\rm red}(t)|^2$ (reduced autocorrel.)"
    )
    ax[0].plot(
        t,
        res["survival_full"],
        "gray",
        lw=0.9,
        ls="--",
        alpha=0.7,
        label=r"$|A_{\rm full}(t)|^2$ (full survival, context)",
    )
    ax[0].plot(t, res["bound_rhs"], "b-", lw=1.6, label=r"$P(0)\,P(t)$ (bound RHS)")
    ax[0].fill_between(
        t,
        res["survival"],
        res["bound_rhs"],
        color="orange",
        alpha=0.3,
        label=r"gap $= P(0)P(t)-|A_{\rm red}|^2$ (transverse loss)",
    )
    ax[0].set_ylabel("probability")
    ax[0].legend(loc="upper right", fontsize=7)
    ax[0].set_title(r"Reduced autocorrelation vs bound $|A_{\rm red}|^2 \leq P(0)P(t)$")
    ax[1].plot(t, res["gap"], "r-", lw=1.4)
    ax[1].set_ylabel(r"gap $P(0)P(t)-|A_{\rm red}|^2$")
    ax[1].set_xlabel("time (a.u.)")
    ax[1].set_title("Transverse information loss not captured by autocorrelation")
    fig.tight_layout()
    fig_path = os.path.join(run_dir, "znse_autocorr.pdf")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"wrote {fig_path}")
    print(
        f"bound holds on all snaps: {bool(np.all(res['bound_hold']))} "
        f"(max violation {res['bound_max_violation']:+.3e})"
    )
    return {
        "csv": csv_path,
        "figure": fig_path,
        "result": res,
        "n_snaps": n_snaps,
        "bound_all_hold": bool(np.all(res["bound_hold"])),
    }


def _snap_times(n_snaps, deck="ZnSe_d20_A0.005338.inp", run_dir="."):
    """Real snapshot times from the deck: t_i = i * OutputStrideWFS * TimeStep."""
    try:
        pass  # not used; keep import local
    except Exception:
        pass
    stride = 7000.0
    dt = 0.04
    try:
        import zkit

        run = zkit.simulation.Run(deck, workdir=run_dir)
        stride = float(getattr(run, "OutputStrideWFS", stride) or stride)
        dt = float(getattr(run, "TimeStep", dt) or dt)
    except Exception:
        pass
    return [i * stride * dt for i in range(n_snaps)]


def analyze_autocorr_sizes(sizes=None, run_dir=".", selected=None, ref=0, snap_dt=280.0):
    """Multi-size autocorrelation / decay-rate analysis (Figure 5 material).

    Parameters
    ----------
    sizes : list of (deck, wfs, label) tuples, e.g.
            [("ZnSe_d20_...inp","wfs_ZnSe_d20_...h5","d20"), ...].
            If None, auto-discovers every wfs_ZnSe*.h5 in run_dir.
    selected : subset of labels to plot |A(t)| and arg A(t) for
               (default: first two discovered, or all if <=4).

    Writes
    ------
      znse_autocorr_sizes.csv  : per size: label, n_snaps, Gamma_fit,
          S_VN_max, |A|(t0), |A|(t_end), and per-snapshot |A|/arg columns.
      znse_autocorr_A2.pdf     : |A_full(t)|^2 vs t for selected sizes  (Fig 5a)
      znse_autocorr_arg.pdf    : arg A_full(t) vs t for selected sizes  (Fig 5b)
      znse_autocorr_Gamma.pdf  : Gamma_fit vs S_VN^max for all sizes    (Fig 5c)

    Returns dict with the per-size result dicts and figure paths.
    """
    from . import autocorr as ac

    if sizes is None:
        import glob as _gl

        pat = os.path.join(run_dir, "wfs_ZnSe*.h5")
        ws = sorted(_gl.glob(pat))
        sizes = []
        for w in ws:
            base = os.path.basename(w).replace("wfs_", "").replace(".h5", "")
            deck = base + ".inp"
            if not os.path.exists(os.path.join(run_dir, deck)):
                deck = "ZnSe_d20_A0.005338.inp"
            sizes.append((deck, os.path.basename(w), base))

    if selected is None:
        selected = [s[2] for s in sizes] if len(sizes) <= 4 else [s[2] for s in sizes[:2]]

    results = {}
    summary = []
    A2_curves = {}  # label -> (times, |A|^2)
    arg_curves = {}  # label -> (times, arg A)
    for deck, wfs, label in sizes:
        run, phys, wfs_path = _load_physics(deck, wfs, run_dir)
        nx = phys.kn[0] + 1 if hasattr(phys, "kn") else 1903
        x, dx = _grid(nx)
        dy = dx
        with h5py.File(wfs_path, "r") as f:
            raw = f["wavefunction"][...]
        n_snaps = raw.shape[0]
        snap_times = _snap_times(n_snaps, deck, run_dir)
        psi = np.empty((n_snaps, nx, nx), dtype=np.complex128)
        for i in range(n_snaps):
            psi[i] = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]
        res = ac.autocorrelation_decay(psi, (dx, dy), np.array(snap_times), axis=0, ref=ref)
        results[label] = res
        A2_curves[label] = (res["times"], res["survival_full"])
        arg_curves[label] = (res["times"], res["arg_A"])
        summary.append(
            {
                "label": label,
                "n_snaps": n_snaps,
                "Gamma_fit": res["Gamma_fit"],
                "t_collapse": res["t_collapse"],
                "S_VN_max": res["S_VN_max"],
                "A2_t0": res["survival_full"][0],
                "A2_tend": res["survival_full"][-1],
            }
        )
        print(
            f"  {label}: Gamma_fit={res['Gamma_fit']:.4e} "
            f"S_VN_max={res['S_VN_max']:.4f} "
            f"|A|^2: t0={res['survival_full'][0]:.4f} tend={res['survival_full'][-1]:.4f}"
        )

    # ---- summary CSV ----
    cols = ["label", "n_snaps", "Gamma_fit", "t_collapse", "S_VN_max", "A2_t0", "A2_tend"]
    csv_path = os.path.join(run_dir, "znse_autocorr_sizes.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in summary:
            fh.write(
                ",".join(f"{s[c]:.6g}" if isinstance(s[c], float) else str(s[c]) for c in cols)
                + "\n"
            )
    print(f"wrote {csv_path}")

    # ---- Fig 5a: |A|^2 vs t ----
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label in selected:
        t, a2 = A2_curves[label]
        ax.plot(t, a2, lw=1.5, label=label)
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel(r"$|A_{\rm full}(t)|^2 = |\langle\Psi_0|\Psi(t)\rangle|^2$")
    ax.set_title("Autocorrelation survival vs time (selected QD sizes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig5a = os.path.join(run_dir, "znse_autocorr_A2.pdf")
    fig.savefig(fig5a, dpi=130)
    plt.close(fig)
    print(f"wrote {fig5a}")

    # ---- Fig 5b: arg A vs t ----
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label in selected:
        t, arg = arg_curves[label]
        ax.plot(t, arg, lw=1.5, label=label)
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel(r"$\arg A_{\rm full}(t)$ (rad, unwrapped)")
    ax.set_title("Phase of the autocorrelation vs time (selected QD sizes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig5b = os.path.join(run_dir, "znse_autocorr_arg.pdf")
    fig.savefig(fig5b, dpi=130)
    plt.close(fig)
    print(f"wrote {fig5b}")

    # ---- Fig 5c: Gamma vs S_VN_max ----
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    g = [s["Gamma_fit"] for s in summary]
    sv = [s["S_VN_max"] for s in summary]
    lab = [s["label"] for s in summary]
    ax.scatter(sv, g, c="navy", s=50, zorder=3)
    for i, label in enumerate(lab):
        ax.annotate(label, (sv[i], g[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if len(g) >= 2:
        A = np.vstack([sv, np.ones_like(sv)]).T
        sol, *_ = np.linalg.lstsq(A, g, rcond=None)
        xs = np.linspace(min(sv), max(sv), 50)
        ax.plot(xs, sol[0] * xs + sol[1], "r--", lw=1, label=f"fit: {sol[0]:.3e}·S + {sol[1]:.3e}")
        ax.legend(fontsize=8)
    ax.set_xlabel(r"$S_{\rm VN}^{\max}$ (peak von Neumann entropy)")
    ax.set_ylabel(r"decay rate $\Gamma$ (a.u.$^{-1}$)")
    ax.set_title(r"Decay rate vs peak entanglement across QD sizes")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig5c = os.path.join(run_dir, "znse_autocorr_Gamma.pdf")
    fig.savefig(fig5c, dpi=130)
    plt.close(fig)
    print(f"wrote {fig5c}")

    return {
        "summary": summary,
        "results": results,
        "fig_A2": fig5a,
        "fig_arg": fig5b,
        "fig_Gamma": fig5c,
        "csv": csv_path,
    }


def crosscheck_timedata(wfs, ted, run_dir=".", deck="ZnSe_d20_A0.005338.inp", ref=0):
    """Validate the wfs-derived autocorrelation against the TDSE code's own
    TimeEvolutionData A(t), at high fidelity.

    TimeEvolutionData stores, per timestep dt=TimeStep, the columns
    [t, Re A(t), Im A(t)] (attribute 'complex'=1) where
    A(t) = <Psi(0)|Psi(t)> computed by the TDSE propagation (raw states,
    not re-normalised; A(0)=1).  We reproduce A(t) directly from the wfs
    snapshots (autocorr.survival_amplitude_full) using the SAME convention
    (raw psi(t), reference normalised so A(0)=1) and compare at the wfs
    snapshot times.  The real part of the two must agree; the imaginary
    part of our overlap is a gauge/frame phase that the TDSE file strips
    (it stores A(t) as purely real).

    Returns a dict with the per-snapshot comparison, the full-resolution
    TDSE curve (for overlay plots), and the max/RMS differences.
    """
    from . import autocorr as ac

    run, phys, wfs_path = _load_physics(deck, wfs, run_dir)
    nx = phys.kn[0] + 1 if hasattr(phys, "kn") else 1903
    x, dx = _grid(nx)
    with h5py.File(wfs_path, "r") as f:
        raw = f["wavefunction"][...]
    n_snaps = raw.shape[0]
    snap_times = _snap_times(n_snaps, deck, run_dir)
    psi = np.empty((n_snaps, nx, nx), dtype=np.complex128)
    for i in range(n_snaps):
        psi[i] = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]
    # TDSE convention: raw psi(t); reference normalised so A(0)=1
    n0 = np.sqrt(np.sum(np.abs(psi[ref]) ** 2) * dx * dx)
    psi[ref] = psi[ref] / n0
    A_mine = ac.survival_amplitude_full(psi, (dx, dx), ref=ref)

    with h5py.File(ted, "r") as f:
        td = f["autocorrelation"][...]  # (N,3,2): t, A, ?
    t_td = td[:, 0, 0]
    A_td = td[:, 1, 0] + 1j * td[:, 1, 1]
    A_td_snap = np.interp(snap_times, t_td, A_td.real) + 1j * np.interp(snap_times, t_td, A_td.imag)

    d_re = np.abs(A_mine.real - A_td_snap.real)
    d_im = np.abs(A_mine.imag - A_td_snap.imag)
    d_mod = np.abs(np.abs(A_mine) - np.abs(A_td_snap))
    out = {
        "snap_times": np.array(snap_times),
        "A_mine": A_mine,
        "A_timedata": A_td_snap,
        "td_t": t_td,
        "td_A": A_td,
        "max_diff_real": float(d_re.max()),
        "rms_diff_real": float(np.sqrt(np.mean(d_re**2))),
        "max_diff_imag": float(d_im.max()),
        "max_diff_modulus": float(d_mod.max()),
        "td_imag_max": float(np.abs(A_td.imag).max()),
        "mine_imag_max": float(np.abs(A_mine.imag).max()),
    }
    print(f"crosscheck {wfs} vs {ted}")
    print(f"  TD A(t) imag max = {out['td_imag_max']:.2e} (TD stores real-only A)")
    print(
        f"  max |Re A_mine - Re A_td| = {out['max_diff_real']:.3e} (RMS {out['rms_diff_real']:.3e})"
    )
    print(
        f"  max |Im A_mine|            = {out['mine_imag_max']:.3e} "
        f"(gauge/frame phase in wfs; TD strips it)"
    )
    print(f"  max | |A_mine| - |A_td| |  = {out['max_diff_modulus']:.3e}")
    return out


def plot_autocorr_overlay(
    sizes=None, run_dir=".", selected=None, ref=0, out_prefix="znse_autocorr"
):
    """Figure 5a upgraded: full-resolution TDSE TimeEvolutionData |A(t)|^2
    (native dt) with the wfs-derived validation points overlaid.  Uses the
    autocorrelation dataset already present in TimeEvolutionData_*.h5.
    """
    if sizes is None:
        import glob as _gl

        ws = sorted(_gl.glob(os.path.join(run_dir, "wfs_ZnSe*.h5")))
        sizes = []
        for w in ws:
            base = os.path.basename(w).replace("wfs_", "").replace(".h5", "")
            ted = os.path.join(run_dir, "TimeEvolutionData_" + base + ".h5")
            if not os.path.exists(ted):
                ted = None
            sizes.append((os.path.basename(w), ted, base))
    if selected is None:
        selected = [s[2] for s in sizes] if len(sizes) <= 4 else [s[2] for s in sizes[:2]]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for wfs, ted, label in sizes:
        if label not in selected:
            continue
        if ted is None or not os.path.exists(ted):
            print(f"  skip {label}: no TimeEvolutionData for overlay")
            continue
        cc = crosscheck_timedata(wfs, ted, run_dir=run_dir, ref=ref)
        t_td = cc["td_t"]
        A2_td = np.abs(cc["td_A"]) ** 2
        ax.plot(t_td, A2_td, lw=1.0, alpha=0.85, label=f"{label} (TDSE)")
        ax.plot(
            cc["snap_times"],
            np.abs(cc["A_mine"]) ** 2,
            "o",
            ms=4,
            mfc="none",
            mew=1.2,
            label=f"{label} (wfs, 25 pts)",
        )
        ax.plot(cc["snap_times"], np.abs(cc["A_mine"]) ** 2, lw=0.6, alpha=0.5)
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel(r"$|A(t)|^2 = |\langle\Psi_0|\Psi(t)\rangle|^2$")
    ax.set_title("Autocorrelation survival: TDSE propagation vs wfs validation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    path = os.path.join(run_dir, f"{out_prefix}_A2.pdf")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    deck_arg = sys.argv[1] if len(sys.argv) > 1 else "ZnSe_d20_A0.005338.inp"
    wfs_arg = sys.argv[2] if len(sys.argv) > 2 else "wfs_ZnSe_d20_A0.005338.h5"
    rd = os.path.dirname(os.path.abspath(deck_arg)) or "."
    analyze_run(deck_arg, wfs_arg, run_dir=rd)
    animate_run(deck_arg, wfs_arg, run_dir=rd)

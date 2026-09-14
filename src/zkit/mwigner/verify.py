"""
zkit.mwigner.verify — rigorous verification of the marginal-Wigner
current pipeline.

Tests the ACTUAL code paths used by analysis.analyze_run /
analysis.animate_run:
  - derivative operators (deriv_x, deriv_p) are spectrally exact
  - current_x = p*W exactly (algebraic)
  - current_p_classical = -V'(x)*W exactly
  - moyal_quantum_force: d_p Q = -R_Q to machine precision, on
    quadratic (Q must be 0) and non-quadratic (V=x^4) potentials
  - full continuity on EXACT analytical states (free particle, HO
    ground, HO coherent): d_x J_x + d_p J_p + d_t W = 0
  - ZnSe real-data self-consistency: closure
    |d_p(J_p^cl - J_p^full) - R_Q| ~ 0 (the paper diagnostic)

run_all() returns (n_pass, n_fail, results) and prints a report.
Every check is a true residual, no hand-waving.
"""

import numpy as np

from . import covariance as cov
from . import current as wc
from . import transform as wq


def _check(results, name, rel, tol):
    ok = bool(rel <= tol)
    results.append((name, rel, tol, ok))
    return ok


def run_all(
    verbose: bool = True,
    deck: str = "ZnSe_d20_A0.005338.inp",
    wfs: str = "wfs_ZnSe_d20_A0.005338.h5",
    run_dir: str = ".",
) -> tuple:
    """Run the full verification suite.

    Parameters
    ----------
    deck, wfs, run_dir : paths for the ZnSe real-data closure test.
        Defaults assume the caller is in the WingerBench directory; pass
        absolute paths to run from elsewhere (e.g. a zkit test).
    """
    results = []
    if verbose:
        print("=== derivative operators (spectral) ===")
    # 0. derivative operators: spectrally exact on a cosine
    N = 256
    L = 20.0
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k0 = 2 * np.pi / L * 3
    f = np.cos(k0 * x)
    df_exact = -k0 * np.sin(k0 * x)
    _check(
        results,
        "deriv_x[cos(kx)] vs -k sin(kx)",
        np.max(np.abs(wc.deriv_x(f, dx) - df_exact)) / np.max(np.abs(df_exact)),
        1e-10,
    )

    # 1. current_x = p*W (algebraic, round-off exact)
    N2 = 64
    X0 = np.linspace(-6, 6, N2)
    P0 = np.linspace(-6, 6, N2)
    W0 = np.exp(-(X0[:, None] ** 2) / 2 - P0[None, :] ** 2) / np.pi
    Jx = wc.current_x(W0, P0)
    _check(
        results,
        "current_x vs p*W",
        np.max(np.abs(Jx - P0[None, :] * W0)) / np.max(np.abs(W0)),
        1e-12,
    )

    # 2. current_p_classical = -V'(x) * W (algebraic)
    Vp = X0
    Jp = wc.current_p_classical(W0, X0, Vp)
    _check(
        results,
        "current_p_classical vs -x*W",
        np.max(np.abs(Jp - (-X0[:, None] * W0))) / np.max(np.abs(W0)),
        1e-12,
    )

    # 3. Moyal Q: quadratic V -> Q == 0 ; non-quadratic -> d_p Q = -R_Q
    V_derivs_q = {"Vp": Vp, "V3": np.zeros_like(X0), "V5": np.zeros_like(X0)}
    Q_q = wc.moyal_quantum_force(W0, X0, P0, V_derivs_q)
    _check(results, "Q == 0 for quadratic V (HO)", np.max(np.abs(Q_q)) / np.max(np.abs(W0)), 1e-10)

    W2 = (1 / np.pi) * np.exp(-(X0[:, None] ** 2) / 2 - P0[None, :] ** 2)
    V_derivs_nq = {"Vp": 4 * X0**3, "V3": 24 * X0, "V5": np.zeros_like(X0)}
    RQ_nq = wc.moyal_residual(W2, X0, P0, V_derivs_nq)
    Q_nq = wc.moyal_quantum_force(W2, X0, P0, V_derivs_nq)
    dp0 = P0[1] - P0[0]
    dQ = wc.deriv_p(Q_nq, dp0)
    _check(
        results, "d_p Q = -R_Q (V=x^4)", np.max(np.abs(dQ + RQ_nq)) / np.max(np.abs(RQ_nq)), 1e-12
    )
    _check(results, "int_p Q dp = 0", np.max(np.abs(Q_nq.sum(axis=1) * dp0)), 1e-10)

    # 4. full continuity on EXACT analytical states (d_t W known)
    if verbose:
        print("=== continuity on exact analytical states ===")
    # widen the grid so displaced Gaussians stay inside the box
    X = np.linspace(-10, 10, 96)
    P = np.linspace(-10, 10, 96)
    dx = X[1] - X[0]
    dp = P[1] - P[0]

    def continuity_free(k):
        sig = 0.15
        Wf = np.exp(-((P[None, :] - k) ** 2) / (2 * sig**2)) / (sig * np.sqrt(2 * np.pi))
        Wf = Wf * np.ones_like(X)[:, None]
        Jxf = wc.current_x(Wf, P)
        Jpf = wc.current_p_classical(Wf, X, np.zeros_like(X))
        dWdt = np.zeros_like(Wf)
        res = wc.deriv_x(Jxf, dx) + wc.deriv_p(Jpf, dp) + dWdt
        return np.max(np.abs(res)) / np.max(np.abs(Wf))

    _check(results, "free particle: continuity (narrow delta)", continuity_free(1.0), 1e-6)
    _check(results, "free particle: continuity (k=2)", continuity_free(2.0), 1e-6)

    def continuity_ho(x0, p0):
        Wc = (1 / np.pi) * np.exp(-((X[:, None] - x0) ** 2) - (P[None, :] - p0) ** 2)
        Jxc = wc.current_x(Wc, P)
        Jpc = wc.current_p_classical(Wc, X, X)
        dWdt = -p0 * wc.deriv_x(Wc, dx) + x0 * wc.deriv_p(Wc, dp)
        res = wc.deriv_x(Jxc, dx) + wc.deriv_p(Jpc, dp) + dWdt
        return np.max(np.abs(res)) / np.max(np.abs(Wc))

    _check(results, "HO ground (x0=0,p0=0): continuity", continuity_ho(0.0, 0.0), 1e-10)
    _check(results, "HO coherent (x0=1,p0=1): continuity", continuity_ho(1.0, 1.0), 1e-10)
    _check(results, "HO coherent (x0=2,p0=-1): continuity", continuity_ho(2.0, -1.0), 1e-10)

    def continuity_ho_moyal(x0, p0):
        Wc = (1 / np.pi) * np.exp(-((X[:, None] - x0) ** 2) - (P[None, :] - p0) ** 2)
        Vdq = {"Vp": X, "V3": np.zeros_like(X), "V5": np.zeros_like(X)}
        Jxc = wc.current_x(Wc, P)
        Jpc = wc.current_p_classical(Wc, X, X)
        Qc = wc.moyal_quantum_force(Wc, X, P, Vdq)
        Jpc_full = Jpc + Qc
        dWdt = -p0 * wc.deriv_x(Wc, dx) + x0 * wc.deriv_p(Wc, dp)
        res = wc.deriv_x(Jxc, dx) + wc.deriv_p(Jpc_full, dp) + dWdt
        return np.max(np.abs(res)) / np.max(np.abs(Wc))

    _check(results, "HO ground + Moyal Q: continuity", continuity_ho_moyal(0.0, 0.0), 1e-10)
    _check(results, "HO coherent + Moyal Q: continuity", continuity_ho_moyal(1.0, 1.0), 1e-10)

    # 5. ZnSe real-data self-consistency (the actual paper diagnostic)
    if verbose:
        print("=== ZnSe real-data closure (current code path) ===")
    from zkit.simulation import Run

    run = Run(run_dir, deck)
    phys = run.physics
    import os

    import h5py

    wfs_path = wfs if os.path.isabs(wfs) else os.path.join(run_dir, wfs)
    with h5py.File(wfs_path, "r") as f:
        raw = f["wavefunction"][...]
    nx = raw.shape[1]
    x = np.linspace(-700, 700, nx)
    dx = x[1] - x[0]
    M_max = nx // 2
    Vp_x = phys.Vpx(x, np.zeros_like(x))
    V3_x = phys.V3_x_slice(x)
    V5_x = phys.V5_x_slice(x)
    V_derivs = {"Vp": Vp_x, "V3": V3_x, "V5": V5_x}
    device = wq.get_device()
    worst = 0.0
    for i in range(raw.shape[0]):
        t_i = phys.snapshot_time(i)
        psi = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]
        W, p_v = wq.get_wigner_research(psi, dx, dx, device=device, M_max=M_max)
        dp = p_v[1] - p_v[0]
        Ax_t = float(phys.Ax(t_i))
        Jx = wc.current_x(W, p_v)
        Jp = wc.current_p_classical(W, x, Vp_x) - Ax_t * W
        Q = wc.moyal_quantum_force(W, x, p_v, V_derivs)
        Jp_q = Jp + Q
        div_diff = wc.deriv_p(Jp - Jp_q, dp)
        RQ = wc.moyal_residual(W, x, p_v, V_derivs)
        Wn = np.max(np.abs(W)) + 1e-300
        rel = np.max(np.abs(div_diff - RQ)) / Wn
        worst = max(worst, rel)
        if i == 0:
            cx_err = np.max(np.abs(Jx - p_v[None, :] * W)) / Wn
            _check(results, "ZnSe: current_x = p*W (snap1)", cx_err, 1e-10)
    _check(results, "ZnSe: closure |d_p(J_p^cl-J_p^full) - R_Q| max over snaps", worst, 1e-4)

    # 6b. 3D marginal Wigner (volume integration) vs analytic 1D Wigner.
    # A 3D separable Gaussian psi(x,y,z)=g(x)g(y)g(z) has x-marginal Wigner
    # equal to the 1D Wigner of g(x), independent of the y,z widths.  This
    # exercises the N-dimensional code path (transverse volume integral).
    if verbose:
        print("=== 3D marginal Wigner vs analytic 1D Wigner ===")
    n3 = 160
    L3 = 40.0
    c3 = np.linspace(-L3 / 2, L3 / 2, n3)
    d3 = c3[1] - c3[0]
    X3, Y3, Z3 = np.meshgrid(c3, c3, c3, indexing="ij")
    s3 = 1.0
    g3 = np.exp(-(X3**2) / (2 * s3**2)) / np.sqrt(np.pi * s3**2)
    psi3 = (
        g3
        * np.exp(-(Y3**2) / (2 * s3**2))
        / np.sqrt(np.pi * s3**2)
        * np.exp(-(Z3**2) / (2 * s3**2))
        / np.sqrt(np.pi * s3**2)
    )
    psi3 = psi3 / np.sqrt(np.sum(np.abs(psi3) ** 2) * d3**3)
    W3, p3 = wq.marginal_wigner(psi3, (d3, d3, d3), axis=0, M_max=40, warn_on_alias=False)
    # analytic 1D Wigner of g(x): W(x,p) = 1/(pi s) exp(-x^2/s^2 - s^2 p^2)
    xx3 = c3[:, None]
    pp3 = p3[None, :]
    W1d = np.exp(-(xx3**2) / s3**2 - s3**2 * pp3**2) / (np.pi * s3)
    werr = np.max(np.abs(W3 - W1d)) / (np.max(np.abs(W1d)) + 1e-300)
    _check(results, "3D marginal Wigner vs analytic 1D Wigner", werr, 5e-2)

    # 6c. 3D vs 2D marginal agreement (separable state must give same x-marginal)
    X2, Y2 = np.meshgrid(c3, c3, indexing="ij")
    psi2 = (np.exp(-(X2**2) / (2 * s3**2)) / np.sqrt(np.pi * s3**2)) * (
        np.exp(-(Y2**2) / (2 * s3**2)) / np.sqrt(np.pi * s3**2)
    )
    psi2 = psi2 / np.sqrt(np.sum(np.abs(psi2) ** 2) * d3**2)
    W2, _ = wq.marginal_wigner(psi2, (d3, d3), axis=0, M_max=40, warn_on_alias=False)
    agree = np.max(np.abs(W3 - W2)) / (np.max(np.abs(W2)) + 1e-300)
    _check(results, "3D vs 2D marginal Wigner agree (separable)", agree, 5e-2)

    # 6. covariance / symplectic-eigenvalue sanity (the legitimate CV diagnostic)
    if verbose:
        print("=== covariance & symplectic eigenvalues ===")
    # pure 2D Gaussian -> all symplectic eigenvalues = 1/2 exactly
    # use a fine grid so the discrete central-difference gradient
    # (O(dx^2)) resolves the momentum variance to ~1e-3
    n_g = 400
    Lg = 20.0
    xg = np.linspace(-Lg / 2, Lg / 2, n_g)
    Xg, Yg = np.meshgrid(xg, xg, indexing="ij")
    dg = xg[1] - xg[0]
    psi_g = np.exp(-(Xg**2 + Yg**2) / 2.0) / np.sqrt(np.pi)
    Sg = cov.compute_4d_covariance(psi_g, dg, dg)
    nu_g = cov.symplectic_eigenvalues(Sg)
    _check(results, "covariance: pure Gaussian symplectic eigs = 1/2", max(abs(nu_g - 0.5)), 1e-10)
    # Gaussian purity of a pure Gaussian state must be exactly 1
    # (tol relaxed to 1e-2: the discrete-gradient ν has ~3e-4 error on
    # this grid, and μ_G = 1/∏(2ν) amplifies it ~8x)
    _check(
        results,
        "covariance: Gaussian purity of pure state = 1",
        abs(cov.gaussian_purity(nu_g) - 1.0),
        1e-6,
    )
    # principal components: symmetric Sigma -> real eigvecs, trace preserved
    pe, pv = cov.principal_components(Sg)
    _check(
        results,
        "covariance: principal-comp trace = Sigma trace",
        abs(np.sum(pe) - np.trace(Sg)),
        1e-9,
    )

    # --- analytic covariance tests (closed-form Gaussian reference) ---
    # (1) correlated 2D Gaussian: psi = N exp(-1/2 r^T M r), M sym PD.
    #     Position block Sigma[x,y] = (M^{-1})/2, momentum block
    #     Sigma[px,py] = M/2 (hbar=1), cross x-p terms exactly 0.
    M = np.array([[2.0, 0.6], [0.6, 1.5]])
    Minv = np.linalg.inv(M)
    n_a = 800
    La = 24.0
    xa = np.linspace(-La / 2, La / 2, n_a)
    Xa, Ya = np.meshgrid(xa, xa, indexing="ij")
    da = xa[1] - xa[0]
    r2 = Xa**2 * M[0, 0] + 2 * Xa * Ya * M[0, 1] + Ya**2 * M[1, 1]
    psi_a = np.exp(-0.5 * r2)
    psi_a = psi_a / np.sqrt(np.sum(np.abs(psi_a) ** 2) * da**2)
    Sa = cov.compute_4d_covariance(psi_a, da, da)
    exp_a = np.zeros((4, 4))
    exp_a[0, 0] = Minv[0, 0] / 2
    exp_a[1, 1] = M[0, 0] / 2
    exp_a[2, 2] = Minv[1, 1] / 2
    exp_a[3, 3] = M[1, 1] / 2
    exp_a[0, 2] = exp_a[2, 0] = Minv[0, 1] / 2
    exp_a[1, 3] = exp_a[3, 1] = M[0, 1] / 2
    _check(
        results,
        "covariance: correlated-Gaussian Sigma vs closed form",
        np.max(np.abs(Sa - exp_a)),
        1e-10,
    )
    # the off-diagonal x-y and px-py correlations are the 'smoking-gun'
    # terms: verify them explicitly against the analytic value
    _check(
        results,
        "covariance: Sigma_x,y vs analytic (Minv_xy/2)",
        abs(Sa[0, 2] - Minv[0, 1] / 2),
        1e-10,
    )
    _check(
        results,
        "covariance: Sigma_px,py vs analytic (M_xy/2, grid-limited)",
        abs(Sa[1, 3] - M[0, 1] / 2),
        1e-10,
    )
    _check(results, "covariance: Sigma_x,px = 0 (real Gaussian)", abs(Sa[0, 1]), 1e-10)

    # (2) HO ground state psi0 = (pi)^-1/4 exp(-x^2/2) (omega=1, hbar=1).
    #     Analytic covariance: Sigma = diag(1/2, 1/2, 1/2, 1/2) and
    #     all x-p cross terms EXACTLY zero (linear-phase Gaussian).
    n_h = 400
    Lh = 30.0
    xh = np.linspace(-Lh / 2, Lh / 2, n_h)
    Xh, Yh = np.meshgrid(xh, xh, indexing="ij")
    dh = xh[1] - xh[0]
    psi_h = np.exp(-(Xh**2 + Yh**2) / 2.0) / np.pi**0.25
    Sh = cov.compute_4d_covariance(psi_h, dh, dh)
    _check(
        results, "covariance: HO ground Sigma = diag(1/2)", np.max(np.abs(np.diag(Sh) - 0.5)), 1e-10
    )
    _check(
        results,
        "covariance: HO ground all off-diag = 0",
        np.max(np.abs(Sh - np.diag(np.diag(Sh)))),
        1e-10,
    )

    # symplectic eigenvalues must never violate the uncertainty principle
    # (nu >= 1/2) for any physical state -- check on the ZnSe snapshots
    nu_min = 1e9
    for i in range(raw.shape[0]):
        psi = raw[i, :, :, 0] + 1j * raw[i, :, :, 1]
        Si = cov.compute_4d_covariance(psi, dx, dx)
        nu_i = cov.symplectic_eigenvalues(Si)
        nu_min = min(nu_min, nu_i.min())
    _check(
        results,
        "covariance: symplectic eigs >= 1/2 (no uncertainty violation)",
        max(0.5 - nu_min, 0.0),
        1e-6,
    )

    # ------------------------------------------------------------------
    # 7. autocorrelation / survival bound |A(t)|^2 <= P(t) = Tr rho(t)^2
    #    (Cauchy-Schwarz for a pure initial state).
    # ------------------------------------------------------------------
    from . import autocorr as ac

    # 7a. Analytic pure product state: |A_red|^2 = P0*P(t) (equality) and
    #     the bound |A_red|^2 <= P0*P(t) holds.  The x-marginal of a
    #     separable Gaussian is mixed (P0 < 1), so the correct check is
    #     |A_red|^2 == P0^2 (not == 1).
    n_p = 200
    Lp = 30.0
    xp = np.linspace(-Lp / 2, Lp / 2, n_p)
    Xp, Yp = np.meshgrid(xp, xp, indexing="ij")
    dp_ = xp[1] - xp[0]
    g = np.exp(-(Xp**2 + Yp**2) / 2.0) / np.pi**0.25  # pure 2D, separable
    series = np.stack([g, g])  # two identical snaps
    r_aa = ac.autocorr_analysis(series, (dp_, dp_), axis=0, times=np.array([0.0, 1.0]))
    P0 = r_aa["P0"]
    _check(
        results,
        "autocorr: |A_red|^2 == P0^2 for pure product state",
        abs(r_aa["survival"][1] - P0**2),
        1e-9,
    )
    _check(
        results,
        "autocorr: bound |A_red|^2<=P0*P holds (trivial)",
        r_aa["bound_max_violation"],
        1e-9,
    )

    # 7b. Real ZnSe data: the rigorous bound |A_red(t)|^2 <= P(0)*P(t) must
    #     hold for EVERY snapshot (no exceptions, including t=0 where it is
    #     an equality).  The gap must be substantially positive during
    #     ionisation -> the autocorrelation alone under-estimates the true
    #     coherence loss to the transverse degree of freedom.
    raw_ac = raw
    psi_ac = np.empty((raw_ac.shape[0], raw_ac.shape[1], raw_ac.shape[2]), dtype=np.complex128)
    for i in range(raw_ac.shape[0]):
        psi_ac[i] = raw_ac[i, :, :, 0] + 1j * raw_ac[i, :, :, 1]
    r_zn = ac.autocorr_analysis(
        psi_ac, (dx, dx), axis=0, times=np.array([i * 280.0 for i in range(raw_ac.shape[0])])
    )
    viol = max(0.0, r_zn["bound_max_violation"])
    _check(results, "autocorr: |A_red|^2<=P0*P bound on all 25 ZnSe snaps", viol, 1e-6)
    _check(
        results,
        "autocorr: gap P0*P-|A_red|^2 substantially positive during ionisation",
        -np.max(r_zn["gap"]),
        -0.1,
    )  # passes iff max gap > 0.1

    n_pass = sum(1 for *_, ok in results if ok)
    n_fail = len(results) - n_pass
    if verbose:
        print()
        for name, rel, tol, ok in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} rel={rel:.3e} (tol {tol:.0e})")
        print(f"\n=== RESULT: {n_pass} PASS, {n_fail} FAIL ===")
        if n_fail:
            print("VERIFICATION FAILED -- do not trust the current figures.")
        else:
            print("ALL CHECKS PASSED -- marginal-Wigner current pipeline is verified.")
    return n_pass, n_fail, results


if __name__ == "__main__":
    run_all()

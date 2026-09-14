"""
zkit.mwigner.autocorr -- autocorrelation / survival analysis and its
rigorous quantum-information bound.

Theoretical background
----------------------
Let rho_x(t) be the *reduced* (x-marginal) density matrix obtained by
tracing the full 2D state |psi(t)><psi(t)| over the transverse
coordinate(s).  Its purity

    P(t) = Tr[ rho_x(t)^2 ]

measures how mixed the x-degree of freedom is (P = 1 iff the x-marginal
state is pure).

The autocorrelation, defined here as the Hilbert-Schmidt overlap of the
reduced density matrices,

    A(t) = Tr[ rho_x(0) rho_x(t) ],

is the rigorously correct "autocorrelation" for the REDUCED space (it
reduces to <psi_x(0)|psi_x(t)> when the reduced state is pure).  By the
Cauchy-Schwarz inequality on the Hilbert-Schmidt inner product,

    |A(t)|^2  <=  Tr[rho_x(0)^2] * Tr[rho_x(t)^2]
             =  P(0) * P(t).

For a PURE INITIAL reduced state (P(0) = 1, i.e. the x-marginal of
|psi(0)> is itself pure) this becomes the clean bound

    |A(t)|^2  <=  P(t)  =  Tr[ rho_x(t)^2 ].

Equality holds iff the reduced state stays pure and unentangled with the
traced-out mode.  The gap

    gap(t)  =  P(0)*P(t) - |A(t)|^2

is the amount of information lost to the transverse degree of freedom
(entanglement / delocalisation) that is NOT captured by the simple
autocorrelation -- exactly the diagnostic the marginal-Wigner covariance
(Sec. covariance) also probes.

IMPORTANT DISTINCTION.  The conventional "survival amplitude"
A_full(t) = <psi(0)|psi(t)> (the FULL 2D overlap) is a different object.
It is bounded by 1 (unitary purity) but is NOT bounded by the
x-marginal purity P(t): a pure separable state has |A_full|^2 = 1 while
P(t) < 1, so |A_full|^2 <= P(t) is false.  The rigorous Cauchy-Schwarz
bound above applies specifically to the REDUCED autocorrelation A(t) =
Tr[rho_x(0) rho_x(t)].  Both quantities are reported; only A_red carries
the bound.

All routines work for an N-dimensional wavefunction; for a 2D state the
transverse coordinate is y, for 3D it is (y, z), etc.  Grid spacings are
taken from `dxs` (one per spatial axis).
"""

import numpy as np


def reduced_density(psi_series, dxs, axis=0):
    """Reduced (axis-marginal) density matrices for every snapshot.

    Returns (rho_series, vol_axis) where
        rho_series[i] = (n_a, n_a) complex matrix,
        rho_series[i][a, b] = (prod_{T} dT) * sum_T psi_i[a,T] psi_i*[b,T],
    and `vol_axis` is the active-axis spacing dx_a (for the trace).
    """
    psi_series = np.asarray(psi_series, dtype=np.complex128)
    ndim = psi_series.ndim - 1
    if np.isscalar(dxs) or (hasattr(dxs, "__len__") and len(dxs) == 1):
        dxs = [float(dxs)] * ndim
    else:
        dxs = [float(d) for d in dxs]
    a = axis % ndim
    dxa = dxs[a]
    vol = float(np.prod(dxs))
    trans_sp = vol / dxa

    rho = np.empty(
        (psi_series.shape[0], psi_series.shape[1], psi_series.shape[1]), dtype=np.complex128
    )
    # build + trace-normalise per snapshot (so a pure reduced state has P=1)
    for i in range(psi_series.shape[0]):
        p = np.moveaxis(psi_series[i], a, 0)  # (n_a, *transverse)
        flat = p.reshape(p.shape[0], -1)  # (n_a, n_trans)
        r = (flat @ np.conj(flat).T) * trans_sp  # (n_a, n_a), unnormalised
        tr = np.trace(r) * dxa
        if abs(tr) > 1e-300:
            r = r / tr
        rho[i] = r
    return rho, dxa


def reduced_purity(psi_series, dxs, axis=0):
    """Reduced (axis-marginal) purity P(t) = Tr[rho_x(t)^2] per snapshot.

    With rho_x trace-normalised (as returned by reduced_density), this is
    simply Tr[rho_x^2] * dx_a^2.
    """
    rho, dxa = reduced_density(psi_series, dxs, axis=axis)
    P = np.empty(rho.shape[0], dtype=np.float64)
    for i in range(rho.shape[0]):
        P[i] = np.real(np.trace(rho[i] @ np.conj(rho[i]).T)) * (dxa**2)
    return P


def autocorrelation(psi_series, dxs, axis=0, ref=0):
    """Reduced autocorrelation A(t) = Tr[ rho_x(ref) rho_x(t) ] (complex).

    This is the Hilbert-Schmidt overlap of the reduced density matrices,
    the rigorously bounded autocorrelation (see module docstring).  The
    conventional full survival amplitude <psi(ref)|psi(t)> is available
    from :func:`survival_amplitude_full`.
    """
    rho, dxa = reduced_density(psi_series, dxs, axis=axis)
    A = np.empty(rho.shape[0], dtype=np.complex128)
    ref_rho = rho[ref]
    for i in range(rho.shape[0]):
        A[i] = np.trace(ref_rho @ rho[i]) * (dxa**2)
    return A


def survival_amplitude_full(psi_series, dxs, ref=0):
    """Conventional full survival amplitude A_full(t) = <psi(ref)|psi(t)>.

    Bounded by 1 under unitary evolution; NOT bounded by the reduced
    purity P(t) -- reported only for context (see module docstring).
    """
    psi_series = np.asarray(psi_series, dtype=np.complex128)
    vol = float(np.prod([dxs] * psi_series.ndim) if np.isscalar(dxs) else np.prod(dxs))
    ref_vec = psi_series[ref].ravel()
    ref_vec = ref_vec / np.sqrt(np.vdot(ref_vec, ref_vec) * vol)
    A = np.empty(psi_series.shape[0], dtype=np.complex128)
    for i in range(psi_series.shape[0]):
        v = psi_series[i].ravel()
        A[i] = np.vdot(ref_vec, v) * vol
    return A


def autocorr_analysis(psi_series, dxs, axis=0, times=None, ref=0):
    """Full reduced-autocorrelation / purity / bound analysis.

    Returns dict with keys:
        times       -- snapshot times
        A           -- reduced autocorrelation Tr[rho(ref) rho(t)] (complex)
        A_full      -- conventional full survival amplitude (complex, context)
        survival    -- |A|^2
        survival_full -- |A_full|^2 (context)
        P           -- P(t) = Tr[rho(t)^2]
        P0          -- P(ref)
        bound_rhs   -- P0 * P(t)
        gap         -- P0*P(t) - |A|^2
        bound_hold  -- |A|^2 <= P0*P(t) (within numerical tolerance)
        bound_max_violation -- largest (|A|^2 - P0*P(t))
    """
    psi_series = np.asarray(psi_series, dtype=np.complex128)
    n = psi_series.shape[0]
    if times is None:
        times = np.arange(n, dtype=float)
    rho, dxa = reduced_density(psi_series, dxs, axis=axis)
    ref_rho = rho[ref]
    A = np.empty(n, dtype=np.complex128)
    P = np.empty(n, dtype=np.float64)
    for i in range(n):
        A[i] = np.trace(ref_rho @ rho[i]) * (dxa**2)
        P[i] = np.real(np.trace(rho[i] @ np.conj(rho[i]).T)) * (dxa**2)
    A_full = survival_amplitude_full(psi_series, dxs, ref=ref)
    P0 = P[ref]
    surv = np.abs(A) ** 2
    surv_full = np.abs(A_full) ** 2
    rhs = P0 * P
    gap = rhs - surv
    tol = 1e-6
    bound_hold = surv <= rhs + tol
    return {
        "times": times,
        "A": A,
        "A_full": A_full,
        "survival": surv,
        "survival_full": surv_full,
        "P": P,
        "P0": P0,
        "bound_rhs": rhs,
        "gap": gap,
        "bound_hold": bound_hold,
        "bound_max_violation": float(np.max(surv - rhs)),
    }


def verify_autocorr_inequality(psi_series, dxs, axis=0, times=None, ref=0):
    """Return (all_hold, max_violation) for the |A|^2 <= P0*P(t) bound."""
    res = autocorr_analysis(psi_series, dxs, axis=axis, times=times, ref=ref)
    return bool(np.all(res["bound_hold"])), res["bound_max_violation"]


def schmidt_entropy(psi_series, dxs, axis=0):
    """Von Neumann entropy S_VN(t) = -sum_i lam_i log lam_i of the reduced
    (axis-marginal) density matrix, where lam_i are its normalised
    eigenvalues (Schmidt eigenvalues). S_VN=0 for a pure reduced state."""
    rho, _ = reduced_density(psi_series, dxs, axis=axis)
    S = np.empty(rho.shape[0], dtype=np.float64)
    for i in range(rho.shape[0]):
        ev = np.linalg.eigvalsh(rho[i])  # rho is Hermitian (real symmetric here)
        ev = np.clip(ev, 0.0, None)
        s = ev.sum()
        if s <= 0:
            S[i] = 0.0
            continue
        lam = ev / s
        lam = lam[lam > 1e-15]
        S[i] = float(-np.sum(lam * np.log2(lam)))
    return S


def autocorrelation_decay(psi_series, dxs, times, axis=0, ref=0):
    """Full survival amplitude A_full(t)=<psi(ref)|psi(t)> plus its decay.

    Returns
    -------
    A_full        : complex overlap series
    survival_full : |A_full|^2  (the quantity to plot)
    arg_A         : unwrapped phase of A_full (radians)
    Gamma_inst    : instantaneous rate Gamma(t) = -d/dt ln|A_full(t)|
                    = -(1/2) d/dt ln|A_full|^2
    Gamma_fit     : exponential decay rate fit over the *early collapse*
                    segment: ln|A_full(t)| = ln|A_full(t_peak)|
                    - Gamma_fit * (t - t_peak), i.e.
                    |A_full(t)| ~ exp(-Gamma_fit * t).  Equivalently
                    Gamma_fit = -(1/2) d/dt ln|A_full|^2 over the window.
                    The window runs from the |A_full|^2 maximum to the first
                    snapshot where |A_full|^2 drops below 1% of that maximum
                    (or the global minimum if the signal never collapses),
                    which isolates the physical ionisation rate and avoids
                    the ill-defined ln(0) floor.
    t_collapse    : time of the 1%-collapse snapshot
    window        : (t_start, t_end, i_start, i_end) used for the fit
    S_VN          : von Neumann entropy of the x-marginal per snapshot
    S_VN_max      : max over snapshots (delocalisation / entanglement proxy)
    """
    A_full = survival_amplitude_full(psi_series, dxs, ref=ref)
    surv = np.abs(A_full) ** 2
    n = len(times)
    # unwrapped phase
    arg_A = np.unwrap(np.angle(A_full))
    # instantaneous rate via central differences on ln survival
    ls = np.log(np.clip(surv, 1e-300, None))
    Gamma_inst = -0.5 * np.gradient(ls, times)

    # --- robust decay-rate estimate -------------------------------------
    # The survival amplitude collapses in an early window then hits the
    # (numerical / ionisation) floor; a single exponential fit over the
    # whole range is ill-defined once the signal reaches ~0.  We therefore
    # fit ln|A|^2 = ln|A(peak)|^2 - Gamma*(t-t_peak) over the *early
    # collapse segment*: from the |A|^2 maximum to the first snapshot where
    # |A|^2 falls below 1% of that maximum (or the global minimum if the
    # signal never collapses).  This isolates the physical ionisation rate.
    i_peak = int(np.argmax(surv))
    a_peak = surv[i_peak]
    thr = 0.01 * a_peak
    i_collapse = i_peak
    for k in range(i_peak + 1, n):
        if surv[k] < thr:
            i_collapse = k
            break
    else:
        # never collapsed below 1% -> use global minimum as the window end
        i_collapse = int(np.argmin(surv))
    i_start = max(i_peak, 1) if i_peak == 0 else i_peak
    i_end = max(i_collapse, i_start + 1)
    tw = times[i_start : i_end + 1]
    lw = ls[i_start : i_end + 1]
    Gamma_fit = float("nan")
    if len(tw) >= 2 and np.ptp(tw) > 0:
        Amat = np.vstack([tw, np.ones_like(tw)]).T
        sol, *_ = np.linalg.lstsq(Amat, lw, rcond=None)
        # lw = ln|A|^2 ; slope = -2*Gamma_fit  => Gamma_fit = -slope/2
        Gamma_fit = -float(sol[0]) / 2.0

    S = schmidt_entropy(psi_series, dxs, axis=axis)
    return {
        "times": times,
        "A_full": A_full,
        "survival_full": surv,
        "arg_A": arg_A,
        "Gamma_inst": Gamma_inst,
        "Gamma_fit": Gamma_fit,
        "t_collapse": float(times[i_collapse]),
        "window": (float(times[i_start]), float(times[i_end]), i_start, i_end),
        "S_VN": S,
        "S_VN_max": float(np.max(S)),
        "arg_at_window": arg_A[i_start : i_end + 1],
    }


__all__ = [
    "reduced_density",
    "reduced_purity",
    "autocorrelation",
    "survival_amplitude_full",
    "autocorr_analysis",
    "verify_autocorr_inequality",
    "schmidt_entropy",
    "autocorrelation_decay",
]

"""
zkit.mwigner.current — the Wigner (phase-space) current computed from
the marginal Wigner function W(x, p_x) produced by
:func:`zkit.mwigner.transform.get_wigner_research`.

The Wigner function obeys a continuity equation in phase space,

    ∂_t W(x, p) + ∂_x J_x(x, p) + ∂_p J_p(x, p) = 0 ,

with the *exact* (approximation-free) x-streaming current

    J_x(x, p) = p * W(x, p) .

For a potential V(x) the p-current is

    J_p(x, p) = -V'(x) * W(x, p)  +  Q(x, p) ,

where Q is the non-local Moyal quantum-force term.  For free motion
(V=0) and for any QUADRATIC potential (e.g. the harmonic oscillator)
the quantum force Q vanishes IDENTICALLY (the Wigner function then obeys
the classical Liouville equation), which makes these the natural
*analytical known solutions* to benchmark against.

The current is therefore as accurate as the Wigner transform itself: J_x
is an exact algebraic product of W with the momentum grid p, and the
only numerical ingredient is the divergence (∂_x, ∂_p), taken with
spectral (FFT) derivatives for machine-precision accuracy on
band-limited states.

The Moyal quantum force Q for a *general* (non-quadratic) potential
is added with :func:`moyal_quantum_force`; see that function.  It is
NOT needed for the analytical benchmarks (free, harmonic oscillator).

NOTE: this module consumes the *marginal* Wigner (y integrated out), so
it recovers only x- and p_x-dependent currents.  A full 4D Wigner
would be needed for y- or p_y-currents.
"""

import math

import numpy as np


# ───────────────────────────── spectral derivatives ─────────────────────────────
def _spectral_deriv_axis(F, dcoord, axis):
    """Exact (to round-off) derivative via FFT for a band-limited,
    domain-localised function on a uniform grid with spacing dcoord.
    """
    n = F.shape[axis]
    k = 2.0j * np.pi * np.fft.fftfreq(n, dcoord)
    shape = [1] * F.ndim
    shape[axis] = n
    k = k.reshape(shape)
    Fk = np.fft.fft(F, axis=axis)
    dF = np.fft.ifft(k * Fk, axis=axis)
    return np.real(dF) if np.isrealobj(F) else dF


def deriv_x(F, dx):
    return _spectral_deriv_axis(F, dx, axis=0)


def deriv_p(F, dp):
    return _spectral_deriv_axis(F, dp, axis=1)


# ───────────────────────────── current components ─────────────────────────────
def current_x(W, p):
    """J_x(x, p) = p * W(x, p)  — exact, no approximation.  p is the 1D
    momentum grid; it broadcasts over the columns of W."""
    p = np.asarray(p, dtype=np.float64)
    return p[None, :] * W


def current_p_classical(W, x, Vp):
    """J_p(x, p) = -V'(x) * W(x, p)  — the classical force term.  Valid
    (exact) for free motion and any quadratic potential (Q=0 there).  Vp = dV/dx(x)."""
    Vp = np.asarray(Vp, dtype=np.float64)
    return -Vp[:, None] * W


def current_p(W, x, Vp=None, moyal_terms=None):
    """Full p-current.  With Vp given, returns the classical force term
    (-V'(x) W).  If `moyal_terms` (a dict of V derivatives, see
    moyal_quantum_force) is supplied, the non-local quantum force is added.
    For free / quadratic V the quantum force is zero, so this equals the
    classical term."""
    Jp = current_p_classical(W, x, Vp) if Vp is not None else np.zeros_like(W)
    if moyal_terms is not None:
        Jp = Jp + moyal_quantum_force(W, x, Vp, moyal_terms)
    return Jp


# ───────────────────────────── Moyal quantum force ─────────────────────────────
def moyal_quantum_force(W, x, p, V_derivs):
    """
    Non-local quantum-force contribution Q(x, p) to J_p for a *general* V(x).

    From the Moyal/Wigner equation (hbar=1, m=1),
        d_t W = -p d_x W + V'(x) d_p W  +  R_Q ,
    where R_Q = moyal_residual(...) is the Moyal odd-derivative series.  In the
    continuity form  d_t W + d_x(p W) + d_p J_p = 0  one identifies
        J_p = -V'(x) W  +  Q ,     with   d_p Q = -R_Q .
    Hence Q is the (spectral) p-antiderivative of -R_Q.  For a p-localised
    Wigner the physically-correct integration constant makes  int_p Q dp = 0
    (the quantum force carries no net p-momentum), so we subtract the p-mean of
    the antiderivative.

    V_derivs: dict {'Vp': dV/dx, 'V3': d3V/dx3, 'V5': d5V/dx5, 'V7': ...}
    (only odd derivatives enter).  Returns Q with the same shape as W.
    """
    dp = p[1] - p[0]
    RQ = moyal_residual(W, x, p, V_derivs)  # (nx, np)
    # spectral antiderivative in p: from  d_p Q = -R_Q  we have
    #   i*k_p * F[Q](k_p) = -F[R_Q](k_p)
    #   =>  F[Q] = -F[R_Q] / (i*k_p) = F[R_Q] / (1j*k_p) .
    # k_p = 2*j*pi*fftfreq is purely imaginary, so (1j*k_p) is real and
    # F[Q] is purely imaginary; the physical real Q is recovered as np.imag
    # of the inverse transform.  The k_p=0 (p-mean) mode is forced to
    # zero: R_Q has zero p-mean for any p-localised Wigner, and forcing it
    # removes the 0/0 at k_p=0 and enforces  int_p Q dp = 0 .
    n = RQ.shape[1]
    kp = 2.0j * np.pi * np.fft.fftfreq(n, dp)  # (np,), purely imaginary
    RQk = np.fft.fft(RQ, axis=1)
    RQk[:, 0] = 0.0  # zero p-mean mode
    with np.errstate(divide="ignore", invalid="ignore"):
        Qk = RQk / (1.0j * kp)[None, :]
    Qk[:, 0] = 0.0  # safety (k_p=0)
    Q = np.imag(np.fft.ifft(Qk, axis=1))
    # enforce int_p Q dp = 0 (remove residual p-mean from round-off)
    Q = Q - Q.mean(axis=1, keepdims=True)
    return Q


def moyal_residual(W, x, p, V_derivs):
    """
    Quantum residual R_Q(x, p) = ∂_t W - (-p ∂_x W + V'(x) ∂_p W) for a general V.

    V_derivs is a dict {'Vp': dV/dx, 'V3': d³V/dx³, 'V5': d⁵V/dx⁵, ...} (only the
    odd derivatives enter; even derivatives vanish by antisymmetry of the Moyal
    bracket).  Implemented with spectral ∂_p derivatives (machine accuracy).

    For free (V=const) and quadratic (V'''=V⁵=...=0) potentials this is EXACTLY
    ZERO, which is the benchmark that proves the classical current is sufficient
    there.  For cubic/quartic/etc. potentials it is non-zero and captures the
    non-locality of the Wigner dynamics.
    """
    dp = p[1] - p[0]
    # guard: derivative arrays must live on the same x-grid as W (axis 0)
    for key in ("Vp", "V3", "V5", "V7"):
        arr = V_derivs.get(key)
        if arr is not None and arr.shape[0] != W.shape[0]:
            raise ValueError(
                f"moyal_residual: {key} has {arr.shape[0]} x-points but W has "
                f"{W.shape[0]}; V_derivs must be sampled on the same x-grid as W"
            )
    # Quantum residual R_Q = the Moyal odd-derivative series
    #   R_Q = Σ_{n≥1} (-1)^n / 2^{2n} / (2n+1)! · V^{(2n+1)}(x) · ∂_p^{2n+1} W .
    # For a given odd order = 2n+1 we have n = (order-1)//2, so the power of 2
    # is 2^{2n} = 2^{order-1} (NOT 2^{order}).
    RQ = np.zeros_like(W)
    for order in (3, 5, 7):
        key = {3: "V3", 5: "V5", 7: "V7"}.get(order)
        if key is None or V_derivs.get(key) is None:
            break
        n = (order - 1) // 2
        coeff = (-1) ** n / (2.0 ** (2 * n)) / math.factorial(order)
        dW = W
        for _ in range(order):
            dW = deriv_p(dW, dp)
        RQ = RQ + coeff * V_derivs[key][:, None] * dW
    return RQ


# ───────────────────────────── continuity residual ─────────────────────────────
def continuity_residual(W, x, p, dWdt, Vp=None, moyal_derivs=None):
    """
    Compute the continuity-equation residual

        R(x, p) = ∂_t W  +  ∂_x(p W)  +  ∂_p J_p ,

    which must be ~0 for a correctly computed current.  dWdt is the analytic (or
    reliably computed) time derivative of W.  Vp = dV/dx for the classical
    p-current; if `moyal_derivs` is given, the Moyal quantum residual is added
    so that the full equation is checked for arbitrary V.

    For the analytical benchmarks (free, harmonic oscillator) Vp is None or the
    HO force, moyal_derivs is None, and R should be ~1e-10 or better.
    """
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    Jx = current_x(W, p)
    Jp = current_p_classical(W, x, Vp) if Vp is not None else np.zeros_like(W)
    div = deriv_x(Jx, dx) + deriv_p(Jp, dp)
    # Naive (classical-force) residual: R_naive = ∂_t W + ∂_x(pW) + ∂_p J_p^cl.
    # The true Moyal continuity equation is R_naive = R_Q (the quantum residual),
    # so the FULL closure residual is  R_naive - moyal_residual, which must be 0.
    R = dWdt + div
    if moyal_derivs is not None:
        R = R - moyal_residual(W, x, p, moyal_derivs)
    return R


# ───────────────────────────── convenience ─────────────────────────────
def compute_current(W, x, p, dWdt=None, Vp=None, moyal_derivs=None):
    """Return a dict with Jx, Jp, and (optionally) the continuity residual."""
    out = {
        "Jx": current_x(W, p),
        "Jp": current_p_classical(W, x, Vp) if Vp is not None else np.zeros_like(W),
    }
    if dWdt is not None:
        out["residual"] = continuity_residual(W, x, p, dWdt, Vp, moyal_derivs)
    return out


def linf_norm(F):
    return np.max(np.abs(F))

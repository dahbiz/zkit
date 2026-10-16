r"""Phase-space (Wigner) current of a marginal Wigner function :math:`W(x, p)`.

:math:`W(x, p)` is the marginal Wigner function of :func:`zkit.mwigner.transform.get_wigner_research`
(the :math:`y` degree of freedom integrated out). It obeys the continuity
equation

.. math::

   \partial_t W + \partial_x J_x + \partial_p J_p = 0, \qquad
   J_x = p\,W, \qquad J_p = -V'(x)\,W + Q(x, p),

where :math:`Q` is the non-local Moyal quantum force (:math:`\hbar = m = 1`).
:math:`Q` vanishes for free motion and for quadratic potentials, where
:math:`W` follows the classical Liouville equation; these cases are exact
references. Derivatives are spectral (FFT) on uniform grids, exact to
round-off for states that decay to zero at the grid boundary. Only the
:math:`x` and :math:`p_x` components are available from the marginal
distribution.
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
    """Spectral derivative along the first (position) axis."""
    return _spectral_deriv_axis(F, dx, axis=0)


def deriv_p(F, dp):
    """Spectral derivative along the second (momentum) axis."""
    return _spectral_deriv_axis(F, dp, axis=1)


# ───────────────────────────── current components ─────────────────────────────
def current_x(W, p):
    r"""Streaming current :math:`J_x(x, p) = p\,W(x, p)`.

    Parameters
    ----------
    W : numpy.ndarray
        Wigner function on the :math:`(x, p)` grid, shape ``(n_x, n_p)``.
    p : array_like
        Momentum grid (length ``n_p``).

    Returns
    -------
    numpy.ndarray
        :math:`J_x` with the shape of ``W``.
    """
    p = np.asarray(p, dtype=np.float64)
    return p[None, :] * W


def current_p_classical(W, x, Vp):
    r"""Classical force current :math:`J_p(x, p) = -V'(x)\,W(x, p)`.

    Exact for free motion and quadratic potentials (:math:`Q = 0`).

    Parameters
    ----------
    W : numpy.ndarray
        Wigner function, shape ``(n_x, n_p)``.
    x : array_like
        Position grid (unused; kept for a uniform signature).
    Vp : array_like
        :math:`V'(x)` on the position grid (length ``n_x``).

    Returns
    -------
    numpy.ndarray
        :math:`J_p` with the shape of ``W``.
    """
    Vp = np.asarray(Vp, dtype=np.float64)
    return -Vp[:, None] * W


def current_p(W, x, Vp=None, moyal_terms=None, p=None):
    """Momentum current :math:`J_p`, optionally including the Moyal quantum force.

    Parameters
    ----------
    W : numpy.ndarray
        Wigner function, shape ``(n_x, n_p)``.
    x : array_like
        Position grid.
    Vp : array_like, optional
        :math:`V'(x)`; without it the classical part is zero.
    moyal_terms : dict, optional
        Odd derivatives of :math:`V` (see :func:`moyal_quantum_force`); adds the
        quantum force :math:`Q`.
    p : array_like, optional
        Momentum grid; required with ``moyal_terms``.

    Returns
    -------
    numpy.ndarray
        :math:`J_p` with the shape of ``W``.

    Raises
    ------
    ValueError
        If ``moyal_terms`` is given without ``p``.
    """
    Jp = current_p_classical(W, x, Vp) if Vp is not None else np.zeros_like(W)
    if moyal_terms is not None:
        if p is None:
            raise ValueError("current_p: the momentum grid p is required with moyal_terms")
        Jp = Jp + moyal_quantum_force(W, x, p, moyal_terms)
    return Jp


# ───────────────────────────── Moyal quantum force ─────────────────────────────
def moyal_quantum_force(W, x, p, V_derivs):
    r"""Moyal quantum force :math:`Q(x, p)` for a general potential.

    With the Moyal residual :math:`R_Q` (:func:`moyal_residual`) the continuity
    form gives :math:`\partial_p Q = -R_Q`. :math:`Q` is obtained as the spectral
    antiderivative in :math:`p`, with the integration constant fixed by
    :math:`\int Q\,dp = 0` (the quantum force carries no net momentum).

    Parameters
    ----------
    W : numpy.ndarray
        Wigner function, shape ``(n_x, n_p)``.
    x : array_like
        Position grid.
    p : array_like
        Uniform momentum grid.
    V_derivs : dict
        Odd derivatives of :math:`V` on the position grid: ``"Vp"``
        (:math:`V'`), ``"V3"``, ``"V5"``, ``"V7"`` (:math:`V^{(3)}`,
        :math:`V^{(5)}`, :math:`V^{(7)}`).

    Returns
    -------
    numpy.ndarray
        :math:`Q` with the shape of ``W``.
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
    r"""Quantum residual of the Moyal equation.

    .. math::

       R_Q = \sum_{n \ge 1} \frac{(-1)^n}{2^{2n} (2n+1)!}\,
             V^{(2n+1)}(x)\,\partial_p^{2n+1} W ,

    i.e. :math:`\partial_t W - (-p\,\partial_x W + V'(x)\,\partial_p W)`. Terms up
    to :math:`V^{(7)}` are included, stopping at the first missing derivative.
    :math:`R_Q = 0` for free motion and quadratic potentials.

    Parameters
    ----------
    W : numpy.ndarray
        Wigner function, shape ``(n_x, n_p)``.
    x : array_like
        Position grid.
    p : array_like
        Uniform momentum grid.
    V_derivs : dict
        ``"V3"``, ``"V5"``, ``"V7"``: odd derivatives of :math:`V` sampled on the
        position grid of ``W``.

    Returns
    -------
    numpy.ndarray
        :math:`R_Q` with the shape of ``W``.

    Raises
    ------
    ValueError
        If a derivative array is not sampled on the position grid of ``W``.
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
    r"""Residual of the phase-space continuity equation.

    .. math::

       R = \partial_t W + \partial_x (p\,W) + \partial_p J_p^{\mathrm{cl}} - R_Q ,

    which vanishes for a correct current. :math:`R_Q` (:func:`moyal_residual`) is
    subtracted only when ``moyal_derivs`` is given.

    Parameters
    ----------
    W : numpy.ndarray
        Wigner function, shape ``(n_x, n_p)``.
    x, p : array_like
        Uniform position and momentum grids.
    dWdt : numpy.ndarray
        Time derivative of :math:`W`.
    Vp : array_like, optional
        :math:`V'(x)` for the classical force current.
    moyal_derivs : dict, optional
        Odd derivatives of :math:`V` for the quantum residual.

    Returns
    -------
    numpy.ndarray
        :math:`R` with the shape of ``W``.
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
    """Phase-space current and, optionally, the continuity residual.

    Parameters
    ----------
    W : numpy.ndarray
        Wigner function, shape ``(n_x, n_p)``.
    x, p : array_like
        Uniform position and momentum grids.
    dWdt : numpy.ndarray, optional
        Time derivative of :math:`W`; adds the continuity residual.
    Vp : array_like, optional
        :math:`V'(x)` for the classical force current.
    moyal_derivs : dict, optional
        Odd derivatives of :math:`V`, used in the residual only.

    Returns
    -------
    dict
        ``Jx`` and ``Jp`` (classical part), and ``residual`` when ``dWdt`` is
        given (:func:`continuity_residual`).
    """
    out = {
        "Jx": current_x(W, p),
        "Jp": current_p_classical(W, x, Vp) if Vp is not None else np.zeros_like(W),
    }
    if dWdt is not None:
        out["residual"] = continuity_residual(W, x, p, dWdt, Vp, moyal_derivs)
    return out


def linf_norm(F):
    """Largest absolute value of an array."""
    return np.max(np.abs(F))

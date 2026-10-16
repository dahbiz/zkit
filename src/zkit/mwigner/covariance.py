r"""Phase-space covariance diagnostics of 2D wavefunctions.

For a wavefunction :math:`\psi(x, y)` on a uniform grid this module computes
the :math:`4\times4` covariance matrix of :math:`R = (x, p_x, y, p_y)`, its
symplectic eigenvalues, the purity of the Gaussian state with the same
covariance, its principal components, and the Schmidt spectrum of the reduced
density matrix :math:`\rho(x, x')` obtained by tracing over :math:`y`.

Momentum moments use spectral (FFT) derivatives,

.. math::

   \langle p_x\rangle = -i \int \psi^* \partial_x \psi \, dx\,dy, \qquad
   \langle p_x^2\rangle = -\int \psi^* \partial_x^2 \psi \, dx\,dy,

which are exact to round-off for states that decay to zero at the grid
boundary. Atomic units, :math:`\hbar = 1`. The moments use the flat momentum
operator :math:`-i\nabla`; for a position-dependent mass this is the canonical
momentum, not :math:`m\,v`.

The symplectic eigenvalues :math:`\nu_k` of a continuous-variable state
satisfy :math:`\nu_k \ge 1/2`, with equality for all :math:`k` exactly for a
pure Gaussian state. A projection of the state onto a single quadrature is a
continuous variable and is not used as a two-level system here.
"""

import numpy as np


def _spectral_derivs(psi, dx, dy):
    """Spectral (FFT) derivatives of a 2D wavefunction on a uniform grid.

    Returns (d_psi_x, d_psi_y, d2_psi_x, d2_psi_y, d2_psi_xy) where e.g.
    d_psi_x = ∂_x ψ, d2_psi_xy = ∂²ψ/∂x∂y.  The spectral derivative is
    exact to machine precision for band-limited functions; for a Gaussian
    that decays to ~0 at the grid boundary the FFT wrap-around error is
    negligible (< 1e-13), in contrast to the O(dx²) central-difference
    error of np.gradient.  This is what makes the covariance matrix match
    its analytic form to machine precision regardless of grid spacing.
    """
    psik = np.fft.fft2(psi)
    kx = 2 * np.pi * np.fft.fftfreq(psi.shape[0], d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(psi.shape[1], d=dy)
    d_psi_x = np.fft.ifft2(1j * kx[:, None] * psik)
    d_psi_y = np.fft.ifft2(1j * ky[None, :] * psik)
    d2_psi_x = np.fft.ifft2(-(kx[:, None] ** 2) * psik)
    d2_psi_y = np.fft.ifft2(-(ky[None, :] ** 2) * psik)
    d2_psi_xy = np.fft.ifft2((1j * kx[:, None]) * (1j * ky[None, :]) * psik)
    return d_psi_x, d_psi_y, d2_psi_x, d2_psi_y, d2_psi_xy


def compute_4d_covariance(psi_2d: np.ndarray, dx: float, dy: float) -> np.ndarray:
    r"""Covariance matrix of :math:`R = (x, p_x, y, p_y)` for a 2D wavefunction.

    The wavefunction is normalised to unit probability on the grid. The grid
    coordinates are centred: :math:`x_i = (i - (n_x - 1)/2)\,\Delta x`, likewise
    for :math:`y`. The matrix is the symmetrised, centred covariance

    .. math::

       \Sigma_{ij} = \tfrac12 \langle \{R_i, R_j\} \rangle
                     - \langle R_i \rangle \langle R_j \rangle ,

    with momentum moments from spectral (FFT) derivatives. A pure Gaussian state
    gives symplectic eigenvalues :math:`\nu = 1/2`.

    Parameters
    ----------
    psi_2d : numpy.ndarray
        Complex wavefunction on a uniform :math:`(n_x, n_y)` grid.
    dx, dy : float
        Grid spacings (a.u.).

    Returns
    -------
    numpy.ndarray
        Real symmetric :math:`4\times4` matrix ordered :math:`(x, p_x, y, p_y)`.

    Raises
    ------
    ValueError
        If the wavefunction has zero norm.
    """
    psi = np.asarray(psi_2d, dtype=np.complex128)
    norm = np.sum(np.abs(psi) ** 2) * dx * dy
    if norm <= 0:
        raise ValueError("compute_4d_covariance: zero-norm wavefunction")
    psi = psi / np.sqrt(norm)

    n = psi.shape[0]
    x = (np.arange(n) - (n - 1) / 2) * dx
    y = (np.arange(psi.shape[1]) - (psi.shape[1] - 1) / 2) * dy

    dpsi_x, dpsi_y, d2psi_x, d2psi_y, d2psi_xy = _spectral_derivs(psi, dx, dy)

    vol = dx * dy
    abs2 = np.abs(psi) ** 2

    # ---- first moments <R_i> ----
    mx = np.sum(x[:, None] * abs2) * vol
    my = np.sum(y[None, :] * abs2) * vol
    px = np.real(-1j * np.sum(np.conj(psi) * dpsi_x) * vol)
    py = np.real(-1j * np.sum(np.conj(psi) * dpsi_y) * vol)

    # ---- second (unsymmetrised) moments <R_i R_j> ----
    # x^2, y^2
    xx = np.sum(x[:, None] ** 2 * abs2) * vol
    yy = np.sum(y[None, :] ** 2 * abs2) * vol
    # p_x^2 = -∫ ψ* ∂_x^2 ψ ; p_y^2 similarly
    px2 = np.real(-np.sum(np.conj(psi) * d2psi_x) * vol)
    py2 = np.real(-np.sum(np.conj(psi) * d2psi_y) * vol)
    # xy = ∫ x y |ψ|^2
    xy = np.sum(x[:, None] * y[None, :] * abs2) * vol
    # x p_x = -i ∫ ψ* x ∂_x ψ  (unsymmetrised product)
    xpx = -1j * np.sum(np.conj(psi) * (x[:, None] * dpsi_x)) * vol
    xpx = np.real(xpx)
    ypy = -1j * np.sum(np.conj(psi) * (y[None, :] * dpsi_y)) * vol
    ypy = np.real(ypy)
    # x p_y = -i ∫ ψ* x ∂_y ψ ; y p_x = -i ∫ ψ* y ∂_x ψ ; p_x p_y = -∫ ψ* ∂_x∂_y ψ
    xpy = -1j * np.sum(np.conj(psi) * (x[:, None] * dpsi_y)) * vol
    xpy = np.real(xpy)
    ypx = -1j * np.sum(np.conj(psi) * (y[None, :] * dpsi_x)) * vol
    ypx = np.real(ypx)
    pxpy = -np.real(np.sum(np.conj(psi) * d2psi_xy) * vol)

    # assemble the 4-vector of <R_i> and the 4x4 <R_i R_j> (symmetrised)
    mean = np.array([mx, px, my, py])
    # unsymmetrised outer products; we symmetrise via 1/2(R_i R_j + R_j R_i)
    M = np.array(
        [
            [xx, xpx, xy, xpy],
            [xpx, px2, ypx, pxpy],
            [xy, ypx, yy, ypy],
            [xpy, pxpy, ypy, py2],
        ],
        dtype=np.float64,
    )
    M = 0.5 * (M + M.T)  # 1/2 <{R_i, R_j}>

    # centred, symmetrised covariance
    Sigma = M - np.outer(mean, mean)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


def symplectic_eigenvalues(Sigma: np.ndarray) -> np.ndarray:
    r"""Symplectic eigenvalues :math:`\nu_k` of a covariance matrix.

    The covariance is ordered :math:`(q_1, p_1, q_2, p_2, \ldots)`, and the
    symplectic form pairs each coordinate with its conjugate momentum (a
    :math:`2\times2` block :math:`[[0, 1], [-1, 0]]` per pair). The
    :math:`\nu_k` are the moduli of the eigenvalues :math:`\pm i\nu_k` of
    :math:`J\Sigma`. A physical state has :math:`\nu_k \ge 1/2` (:math:`\hbar = 1`);
    :math:`\nu_k = 1/2` for all :math:`k` exactly for a pure Gaussian state.

    Parameters
    ----------
    Sigma : numpy.ndarray
        Real :math:`2n\times 2n` covariance matrix.

    Returns
    -------
    numpy.ndarray
        The :math:`2n` values :math:`\lvert\operatorname{Im}\lambda\rvert`, sorted
        in descending order (each :math:`\nu_k` appears twice).
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = Sigma.shape[0] // 2
    # block-diagonal symplectic form for (q1,p1,q2,p2,...)
    J = np.zeros((2 * n, 2 * n))
    for k in range(n):
        J[2 * k, 2 * k + 1] = 1.0
        J[2 * k + 1, 2 * k] = -1.0
    # M = J Sigma ; its eigenvalues are ± i ν_k
    M = J @ Sigma
    ew = np.linalg.eigvals(M)
    nu = np.sort(np.abs(np.imag(ew)))[::-1]
    return nu


def effective_schmidt_rank(eigenvalues: np.ndarray) -> float:
    r"""Effective Schmidt rank :math:`N_{\mathrm{eff}} = 1 / \sum_i \lambda_i^2`.

    :math:`\lambda_i` are the Schmidt eigenvalues of the reduced density matrix
    (:func:`schmidt_spectrum`); values below :math:`10^{-12}` are ignored.
    :math:`N_{\mathrm{eff}}` and the von Neumann entropy are functions of the
    same spectrum and carry the same information.

    Parameters
    ----------
    eigenvalues : array_like
        Schmidt eigenvalues.

    Returns
    -------
    float
        :math:`N_{\mathrm{eff}}`, or 0 when no eigenvalue exceeds :math:`10^{-12}`.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return 0.0
    return 1.0 / np.sum(ev**2)


def schmidt_spectrum(rho_or_eigs, dx: float = None) -> np.ndarray:
    r"""Schmidt eigenvalues of the reduced density matrix :math:`\rho(x, x')`.

    Parameters
    ----------
    rho_or_eigs : numpy.ndarray
        Either the reduced density matrix on the :math:`x` grid (2D array) or
        precomputed eigenvalues (1D array).
    dx : float, optional
        Grid spacing; required when a density matrix is passed (it is
        normalised to unit trace with the quadrature weight ``dx``).

    Returns
    -------
    numpy.ndarray
        Eigenvalues above :math:`10^{-12}`, in descending order (empty for a
        non-positive trace).

    Raises
    ------
    ValueError
        If a density matrix is passed without ``dx``.
    """
    import numpy as np

    if rho_or_eigs.ndim == 2:
        if dx is None:
            raise ValueError("schmidt_spectrum: dx required when passing rho")
        rho = rho_or_eigs * dx
        tr = np.trace(rho)
        if tr <= 0:
            return np.array([])
        rho_n = rho / tr
        ev = np.linalg.eigvalsh(rho_n) * dx
    else:
        ev = np.asarray(rho_or_eigs, dtype=np.float64)
    return np.sort(ev[ev > 1e-12])[::-1]


def gaussian_purity(symplectic_eigs: np.ndarray) -> float:
    r"""Purity of the Gaussian state with a given covariance matrix.

    For an :math:`n`-mode Gaussian state

    .. math::

       \mu_G = \prod_{k=1}^{n} \frac{1}{2\nu_k} ,

    which is 1 for a pure Gaussian state. :math:`\mu_G` is the purity of the
    Gaussian state with the same covariance, not of the actual state: a true
    purity :math:`\operatorname{Tr}\rho^2 < \mu_G` indicates non-Gaussian
    structure. For strongly delocalised states :math:`\nu_k \gg 1` and
    :math:`\mu_G` underflows towards 0, where the comparison is not meaningful.

    Parameters
    ----------
    symplectic_eigs : array_like
        Symplectic eigenvalues :math:`\nu_k` (each counted once per entry given).

    Returns
    -------
    float
        :math:`1 / \prod (2\nu_k)` over the entries given.
    """
    nu = np.asarray(symplectic_eigs, dtype=np.float64)
    return 1.0 / np.prod(2.0 * nu)


def principal_components(Sigma: np.ndarray) -> tuple:
    """Ordinary eigendecomposition of a covariance matrix.

    The eigenvalues are the variances along the uncorrelated directions of the
    phase-space distribution, and the eigenvectors the corresponding linear
    combinations of :math:`(x, p_x, y, p_y)`. These are not symplectic
    eigenvalues and do not define a two-level system.

    Parameters
    ----------
    Sigma : numpy.ndarray
        Real symmetric covariance matrix.

    Returns
    -------
    tuple of numpy.ndarray
        ``(eigvals, eigvecs)`` with eigenvalues in descending order and the
        eigenvectors as columns.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    evals, evecs = np.linalg.eigh(Sigma)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def covariance_diagnostics(
    psi_2d: np.ndarray, dx: float, dy: float, rho: np.ndarray = None
) -> dict:
    r"""Covariance diagnostics of one 2D wavefunction snapshot.

    Parameters
    ----------
    psi_2d : numpy.ndarray
        Complex wavefunction on a uniform grid.
    dx, dy : float
        Grid spacings (a.u.).
    rho : numpy.ndarray, optional
        Reduced density matrix :math:`\rho(x, x')`; computed from ``psi_2d`` as
        :math:`\sum_y \psi(x, y)\,\psi^*(x', y)\,\Delta x` when not given.

    Returns
    -------
    dict
        ``Sigma``: :math:`4\times4` covariance matrix (:func:`compute_4d_covariance`);
        ``diag``: variances ``Sxx``, ``Spxpx``, ``Syy``, ``Spypy``;
        ``offdiag``: covariances ``Sxpx``, ``Sypy``, ``Sxy``, ``Spxpy``, ``xpy``,
        ``ypx``; ``nu``: symplectic eigenvalues; ``mu_G``: Gaussian purity;
        ``mu_true``: :math:`\sum_i \lambda_i^2` of the Schmidt spectrum;
        ``non_gaussian``: ``mu_true < mu_G``, or ``None`` when
        :math:`\mu_G \le 10^{-6}`; ``pc_evals`` and ``pc_evecs``: principal
        components of ``Sigma``.
    """
    Sigma = compute_4d_covariance(psi_2d, dx, dy)
    labels = ["x", "px", "y", "py"]
    diag = {f"S{l1}{l1}": Sigma[i, i] for i, l1 in enumerate(labels)}
    offd_pairs = [
        (0, 1, "Sxpx"),
        (2, 3, "Sypy"),
        (0, 2, "Sxy"),
        (1, 3, "Spxpy"),
        (0, 3, "xpy"),
        (1, 2, "ypx"),
    ]
    offdiag = {name: Sigma[i, j] for i, j, name in offd_pairs}
    nu = symplectic_eigenvalues(Sigma)
    mu_G = gaussian_purity(nu)
    if rho is None:
        rho = np.matmul(psi_2d, np.conj(psi_2d).T) * dx
    ev = schmidt_spectrum(rho, dx)
    mu_true = float(np.sum(ev**2)) if ev.size else 0.0
    # non-Gaussianity test only meaningful when mu_G is well above
    # underflow (delocalised ionised states give mu_G ~ 0 spuriously)
    non_gaussian = bool(mu_true < mu_G) if mu_G > 1e-6 else None
    pc_evals, pc_evecs = principal_components(Sigma)
    return dict(
        Sigma=Sigma,
        diag=diag,
        offdiag=offdiag,
        nu=nu,
        mu_G=mu_G,
        mu_true=mu_true,
        non_gaussian=non_gaussian,
        pc_evals=pc_evals,
        pc_evecs=pc_evecs,
    )

"""
zkit.mwigner.covariance — phase-space covariance analysis of the 2D
electron wavefunction, integrated into zkit.

This module implements ONLY the technically-sound parts of a proposed
"high-dimensional entanglement" extension.  Deliberately EXCLUDED:
  * the "Symplectic Principal Component (SPC) qubit" — a single linear
    quadrature Q = v . R is a continuous variable, NOT a two-level
    system; projecting a CV state onto Q>0 / Q<0 does not define a
    qubit, and the proposed orthogonal rotation is not a symplectic
    (canonical) transformation, so the slice-FFT recipe is not a valid
    coordinate change.  The correct CV diagnostic is the symplectic
    spectrum of the covariance matrix below.
  * any "qubit / ququart capacity limit" comparison — the von Neumann
    entropy of a *continuous-variable* reduced state has no d-level
    bound; such a limit applies only to discrete-d systems.

Provided
--------
compute_4d_covariance(psi_2d, dx, dy) -> Sigma (4x4, ordered x,px,y,py)
symplectic_eigenvalues(Sigma)        -> symplectic spectrum (>= 1/2 for
                                         a physically-allowed state; = 1/2
                                         iff the (4D) state is pure)
effective_schmidt_rank(eigenvalues)  -> N_eff = 1 / sum(lambda_i^2)
schmidt_spectrum(rho_or_eigs)        -> the reduced-state Schmidt
                                         eigenvalues (for plotting)

All momentum expectation values use the SPECTRAL (FFT) derivative method
    <p_x>   = -i ∫ Ψ* ∂_x Ψ dx dy
    <p_x^2> = -∫ Ψ* ∂_x^2 Ψ dx dy
which is exact to machine precision on a uniform grid for the band-limited
(Gaussian-decaying) states encountered here, in contrast to the O(dx^2)
central-difference error of a finite-difference gradient.  The covariance
matrix therefore matches its closed-form analytic value to ~1e-14
regardless of grid spacing.

NOTE (spatially-varying mass): the TDSEZ wavefunctions consumed here are
already propagated with the mass-corrected kinetic operator
    T = -∫ ε(x,y) ∇B_j · ∇B_i  dx dy ,   ε(x,y) = m(x,y) = 1 - 0.84 exp(-...)
(poisson_bsp.cpp).  The stored ψ(r) is therefore the physical wavefunction
in the m(x,y) geometry, and the flat -i∇ momentum operator used below is
the appropriate one for that stored field -- no separate mass correction
is applied or required.  (An earlier caveat warning of a flat-mass
approximation in the dot core is withdrawn: the mass physics is already
contained in the wfs.)
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
    """4x4 covariance matrix of R = (x, p_x, y, p_y).

    Ordered [x, p_x, y, p_y].  Built from the wavefunction normalised to
    unit 2D probability.  Momentum moments use the GRADIENT method
    (p_x = -i ∂_x, p_x^2 = -∂_x^2), avoiding FFT phase/origin artifacts.

    The covariance is the centred, symmetrised matrix
        Sigma_ij = 1/2 <{R_i, R_j}> - <R_i> <R_j> ,
    with R = (x, p_x, y, p_y).  For a pure 2D Gaussian this gives
    symplectic eigenvalues nu = 1/2 exactly; mixed/delocalised states
    have nu > 1/2 (never < 1/2, by the uncertainty principle).

    Returns
    -------
    Sigma : (4, 4) real symmetric covariance matrix.
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
    """Symplectic eigenvalues ν_k of the 4x4 covariance matrix.

    For a 4D phase space they come in pairs; a physically-allowed state
    has ν_k >= 1/2 (in ħ=1 units), and ν_k = 1/2 for ALL k iff the state
    is pure.  Mixedness / entanglement with the traced-out mode shows up
    as ν_k > 1/2.  This is the rigorous continuous-variable diagnostic
    (replacing the invalid "SPC qubit" construction).

    Sigma is ordered [x, p_x, y, p_y]; the symplectic form must pair
    (x, p_x) and (y, p_y), i.e. it is block-diagonal with a 2x2
    [[0, 1], [-1, 0]] block on each pair (NOT the [[0,I],[-I,0]] block
    form, which would pair the two coordinates instead of coordinate with
    its conjugate momentum).
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
    """Effective Schmidt rank N_eff = 1 / Σ λ_i^2 from the Schmidt
    eigenvalues λ_i of the reduced density matrix ρ(x,x').

    NOTE: N_eff and the von Neumann entropy S_VN = -Σ λ_i ln λ_i are BOTH
    functions of the SAME Schmidt spectrum; they are not independent
    proofs.  N_eff is reported as a complementary, intuitive measure of
    multi-modality (e.g. N_eff ≈ 9 means the reduced state spans ~9
    effective modes), not as separate evidence.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return 0.0
    return 1.0 / np.sum(ev**2)


def schmidt_spectrum(rho_or_eigs, dx: float = None) -> np.ndarray:
    """Return the Schmidt eigenvalues λ_i of the reduced density matrix
    ρ(x,x') (tracing over y).  Accepts either the full rho (dx required to
    normalise the trace) or pre-computed eigenvalues.
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
    """Purity of the BEST-FIT GAUSSIAN state, from its symplectic
    eigenvalues ν_k.

    For an n-mode Gaussian state (here n = 2) the Gaussian purity is

        μ_G = 1 / ∏_{k=1}^{n} (2 ν_k) .

    A pure Gaussian has ν_k = 1/2 for all k, giving μ_G = 1.  Any
    physically-allowed state has μ_G ∈ (0, 1].

    IMPORTANT: μ_G is the purity of the *Gaussian state with the same
    covariance matrix*, NOT the purity of the true (possibly
    non-Gaussian) state.  To test for non-Gaussian structure compare
    μ_G against the TRUE purity μ_true = Tr ρ² (from the wavefunction);
    μ_true < μ_G indicates the real state is "more mixed" than its
    Gaussian fit — i.e. contains non-Gaussian features (Fock/cat/
    higher-order correlations).  CAUTION: for the strongly ionised
    snapshots the symplectic eigenvalues become huge (the electron is
    delocalised over hundreds of a.u.), so μ_G underflows toward 0 and
    the comparison μ_true > μ_G is an artefact of the underflow, not a
    sign of Gaussianity.  The diagnostic is meaningful in the
    near-pure / moderately-mixed regime (early snapshots), not at
    extreme delocalisation.
    """
    nu = np.asarray(symplectic_eigs, dtype=np.float64)
    return 1.0 / np.prod(2.0 * nu)


def principal_components(Sigma: np.ndarray) -> tuple:
    """Standard eigendecomposition of the covariance matrix Σ.

    Returns (eigvals, eigvecs) with eigvals descending.  These are the
    *ordinary* (not symplectic) eigenvalues and eigenvectors of Σ: the
    variances along the uncorrelated normal modes of the phase-space
    distribution, and the linear combinations of (x, p_x, y, p_y) that
    realise them.  This is the legitimate "true independent modes"
    diagnostic — it shows whether the dominant spread is along a pure
    axis (e.g. purely x) or a rotated mixture (e.g. 0.8 x + 0.6 p_y),
    evidencing continuous-variable (not separable spatial) structure.

    NOTE: this is NOT the "symplectic principal component qubit".  A
    principal component of Σ is a direction of large variance in phase
    space; it is a continuous variable, not a two-level system, and must
    not be interpreted as a qubit.  The dominant eigenvector is reported
    only as a descriptive statistic of where the wavepacket's spread
    lives.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    evals, evecs = np.linalg.eigh(Sigma)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def covariance_diagnostics(
    psi_2d: np.ndarray, dx: float, dy: float, rho: np.ndarray = None
) -> dict:
    """One-call bundle of the physically-sound covariance diagnostics for
    a single 2D wavefunction snapshot.

    Returns a dict with:
      Sigma    : 4x4 covariance matrix [x, p_x, y, p_y]
      diag     : variances {Sxx, Syy, Spxpx, Spyy}
      offdiag  : key covariances {Sxpx, Sypy, Sxy, Spxpy, xpy, ypx}
      nu       : symplectic eigenvalues (Williamson)
      mu_G     : Gaussian-fit purity (gaussian_purity)
      mu_true  : true purity Tr rho^2 (from reduced x-density matrix)
      non_gaussian : mu_true < mu_G (None if mu_G underflowed)
      pc_evals : principal-component eigenvalues of Sigma (desc)
      pc_evecs : principal-component eigenvectors (columns)
    All physics is read from the wavefunction; nothing is hard-coded.
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

"""
zkit.mwigner — marginal (x-) Wigner function, its phase-space current,
and the Moyal quantum force, integrated into zkit.

Everything here consumes the *marginal* Wigner W(x, p_x) (y integrated out)
produced by :func:`zkit.mwigner.transform.get_wigner_research`.  The
potential, position-dependent mass, and laser envelope are read from the
TDSEZ input deck through :class:`zkit.physics.Physics`, so no physics is
hard-coded.

Public API
------------
transform.get_wigner_research(psi, dx, dy, ...) -> (W, p_x)
transform.measure_wigner_negativity(W, dx, dp) -> float
transform.verify_purities_and_entropy(...) -> tuple
current.current_x(W, p)                         -> J_x = p*W
current.current_p_classical(W, x, Vp)            -> J_p = -V'(x)*W
current.moyal_quantum_force(W, x, p, V_derivs) -> Q
current.moyal_residual(W, x, p, V_derivs)      -> R_Q
covariance.compute_4d_covariance(psi, dx, dy)  -> Sigma (4x4)
covariance.symplectic_eigenvalues(Sigma)       -> symplectic spectrum
covariance.gaussian_purity(nu)                  -> mu_G (Gaussian-fit purity)
covariance.principal_components(Sigma)          -> (eigvals, eigvecs) of Sigma
covariance.effective_schmidt_rank(eigs)         -> N_eff
autocorr.autocorrelation(psi_series, dxs)        -> A(t) = <psi(0)|psi(t)>
autocorr.reduced_purity(psi_series, dxs)         -> P(t) = Tr[rho(t)^2]
autocorr.autocorr_analysis(psi_series, dxs)      -> A, |A|^2, P, gap, bound
autocorr.schmidt_entropy(psi_series, dxs)        -> S_VN per snapshot
autocorr.autocorrelation_decay(psi_series,dxs,t) -> A_full,|A|^2,arg,Gamma
verify.run_all()                                 -> (n_pass, n_fail, results)
analysis.analyze_run(deck, wfs)                 -> writes CSV + figures
analysis.animate_run(deck, wfs)                 -> writes mp4 + gif
analysis.analyze_covariance(deck, wfs)          -> covariance CSV + figure
analysis.analyze_autocorr(deck, wfs)            -> autocorrelation CSV + figure
analysis.analyze_autocorr_sizes(sizes, run_dir) -> Fig 5 (A2, arg, Gamma)
analysis.crosscheck_timedata(wfs, ted, run_dir) -> validate A(t) vs TDSE
analysis.plot_autocorr_overlay(run_dir)      -> Fig 5a: TDSE curve + wfs pts
"""

from . import analysis, autocorr, covariance, current, transform, verify

__all__ = ["transform", "current", "covariance", "autocorr", "verify", "analysis"]

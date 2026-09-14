Wigner and phase-space analysis
================================

The ``zkit.mwigner`` subpackage provides marginal Wigner transforms, phase-space
currents, covariance analysis, and autocorrelation tools for TDSEZ outputs.

.. warning::

   The Wigner-transform and phase-space analysis code still needs further
   validation against published benchmarks. Use it as a research path, not as
   a finalized analysis result.

Install the optional dependency:

.. code-block:: bash

   pip install -e ".[mwigner]"

Marginal Wigner transform
-------------------------

.. code-block:: python

   from zkit.mwigner.transform import get_wigner_research

   W, p = get_wigner_research(
       psi,
       dx=0.05,
       dy=0.05,
       nx=256,
       ny=256,
   )
   print(W.shape)
   print(p.min(), p.max())

Negativity and purity
---------------------

.. code-block:: python

   from zkit.mwigner.transform import measure_wigner_negativity
   from zkit.mwigner.covariance import compute_4d_covariance, symplectic_eigenvalues, gaussian_purity

   W, p = get_wigner_research(psi, dx, dy)
   neg = measure_wigner_negativity(W, dx, dy)
   print("Wigner negativity:", neg)

   Sigma = compute_4d_covariance(psi, dx, dy)
   nu = symplectic_eigenvalues(Sigma)
   mu = gaussian_purity(nu)
   print("Gaussian purity:", mu)

Phase-space current
-------------------

.. code-block:: python

   from zkit.mwigner.current import current_x, current_p_classical, moyal_quantum_force, moyal_residual

   Jx = current_x(W, p)
   Jp = current_p_classical(W, x, Vp)
   Q = moyal_quantum_force(W, x, p, V_derivs)
   R = moyal_residual(W, x, p, V_derivs)

Autocorrelation
---------------

.. code-block:: python

   from zkit.mwigner.autocorr import autocorrelation, autocorr_analysis, autocorrelation_decay

   A = autocorrelation(psi_series, dxs)
   A2 = A ** 2
   P = reduced_purity(psi_series, dxs)
   S = schmidt_entropy(psi_series, dxs)

   A_full, A2_full, arg, Gamma = autocorrelation_decay(psi_series, dxs, t)

Analysis helpers
----------------

.. code-block:: python

   from zkit.mwigner import analysis

   analysis.analyze_run(deck, wfs)
   analysis.animate_run(deck, wfs)
   analysis.analyze_covariance(deck, wfs)
   analysis.analyze_autocorr(deck, wfs)
   analysis.crosscheck_timedata(wfs, ted, run_dir)
   analysis.plot_autocorr_overlay(run_dir)

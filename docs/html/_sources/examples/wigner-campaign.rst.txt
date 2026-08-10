Wigner campaign
================

This example turns a set of time-dependent snapshots into a phase-space
coherence analysis: marginal Wigner functions, purity, current, and
autocorrelation decay.

Data
----

::

    /scratch/cavity_qed/
      td/
        wfs_cavity_1.h5
        TimeEvolutionData_cavity_1.h5

1. Load snapshots
-----------------

.. code-block:: python

   from zkit.io.wavefunction import read_wfs

   wfs = read_wfs("/scratch/cavity_qed/td/wfs_cavity_1.h5")
   print("snapshots:", wfs.n_snapshots)
   print("dof:", wfs.data.shape[1])
   print("time spacing is implied by stride:", wfs.times[:5])

2. Reconstruct each snapshot on a phase-space grid
--------------------------------------------------

.. code-block:: python

   from zkit.io.wfs_field import reconstruct_wfs

   nx, ny = 300, 300
   snapshots = []
   for step in range(wfs.n_snapshots):
       res = reconstruct_wfs(
           "/scratch/cavity_qed/td/wfs_cavity_1.h5",
           step=step,
           npoints=nx if wfs.dim == 1 else (nx, ny),
       )
       snapshots.append(res["psi"])

3. Compute marginal Wigner function
------------------------------------

.. code-block:: python

   from zkit.mwigner.transform import get_wigner_research

   W, p = get_wigner_research(
       snapshots[0],
       dx=0.02,
       dy=0.02,
       nx=256,
       ny=256,
   )
   print("W shape:", W.shape)

4. Measure nonclassicality
---------------------------

.. code-block:: python

   from zkit.mwigner.transform import measure_wigner_negativity
   from zkit.mwigner.covariance import compute_4d_covariance, symplectic_eigenvalues, gaussian_purity

   neg = measure_wigner_negativity(W, dx=0.02, dy=0.02)
   print("negativity:", neg)

   Sigma = compute_4d_covariance(snapshots[0], dx=0.02, dy=0.02)
   nu = symplectic_eigenvalues(Sigma)
   print("symplectic eigenvalues:", nu)
   print("Gaussian purity:", gaussian_purity(nu))

5. Phase-space current
----------------------

.. code-block:: python

   from zkit.mwigner.current import current_x, moyal_quantum_force

   x = np.linspace(-3, 3, 256)
   Jx = current_x(W, p)
   Q = moyal_quantum_force(W, x, p, V_derivs=(Vpp, Vppp, Vppppp))

6. Autocorrelation decay
------------------------

.. code-block:: python

   from zkit.mwigner.autocorr import autocorrelation, autocorrelation_decay
   import numpy as np

   dxs = [0.02, 0.02]
   A = autocorrelation(snapshots, dxs)
   A_full, A2, arg, Gamma = autocorrelation_decay(snapshots, dxs, t)

   t = np.arange(len(snapshots))
   plt.figure()
   plt.plot(t, np.abs(A) ** 2)
   plt.xlabel("snapshot")
   plt.ylabel("|A(t)|^2")
   plt.tight_layout()
   plt.savefig("figures/cavity_autocorr.png", dpi=150)

7. Campaign report
------------------

.. code-block:: python

   from zkit.mwigner import analysis

   analysis.analyze_run(deck="/scratch/cavity_qed/cavity_1.inp", wfs=snapshots)
   analysis.animate_run(deck="/scratch/cavity_qed/cavity_1.inp", wfs=snapshots)
   analysis.analyze_covariance(deck="/scratch/cavity_qed/cavity_1.inp", wfs=snapshots)
   analysis.analyze_autocorr(deck="/scratch/cavity_qed/cavity_1.inp", wfs=snapshots)

Python API
==========

zkit exposes three main layers:

1. High-level ``Run`` object that loads an entire simulation directory.
2. Low-level readers in ``zkit.io`` for individual HDF5 outputs.
3. Visualization and analysis helpers in ``zkit.viz`` and ``zkit.mwigner``.

Top-level shortcuts
-------------------

.. code-block:: python

   import zkit

   # Convenience aliases exported from zkit.__init__
   zkit.open_eig
   zkit.reconstruct_knots
   zkit.partition_of_unity
   zkit.infinite_well_energy
   zkit.find_binary
   zkit.run
   zkit.Run
   zkit.load
   zkit.read_eigen
   zkit.read_evolution
   zkit.read_timeseries
   zkit.read_wfs
   zkit.read_tdm
   zkit.tdm_of_state
   zkit.tdm_magnitude_of_state
   zkit.reconstruct_wfs
   zkit.reconstruct_static_wfs
   zkit.wfs_to_vtk
   zkit.compute_wfs_norm
   zkit.detect_dimension
   zkit.plot_wavefunction
   zkit.plot_transition_diagram
   zkit.plot_tdm_matrix
   zkit.plot_tdm

Load a whole run
----------------

.. code-block:: python

   from zkit.simulation import Run

   run = Run("run_dir", "h2p.inp")
   print("dimension:", run.dim)
   print("n_steps:", run.evolution.n_steps)
   print("spectrum:", run.spectrum.values)
   print("time:", run.time[:5])
   print("dipoles:", run.dipoles.shape)
   print("populations:", run.populations.shape)
   print("energies:", run.energies.shape)
   print("currents:", run.currents.shape)
   print("autocorrelation:", run.autocorrelation.shape)
   print("ts times:", run.ts.times)

The ``Run`` object parses the input file when available and falls back to
data-derived metadata otherwise.

Read individual files
---------------------

.. code-block:: python

   from zkit.io.eigen import read_eigen
   from zkit.io.evolution import read_evolution
   from zkit.io.ts import read_timeseries
   from zkit.io.wavefunction import read_wfs
   from zkit.io.tdm import read_tdm, tdm_of_state, tdm_magnitude_of_state
   from zkit.io.base import detect_dimension

   # Eigenvalues and eigenvectors
   eigen = read_eigen("static/EigenData_h2p.inp.h5")
   print(eigen.values.shape)
   print(eigen.vectors.shape)

   # Time evolution observables
   evolution = read_evolution("td/TimeEvolutionData_h2p.inp.h5")
   print(evolution.dimension)

   # PETSc TS snapshots
   ts = read_timeseries("ts_h2p.inp.h5")
   print(ts.times)

   # Wavefunction snapshots
   wfs = read_wfs("td/wfs_h2p.inp.h5")
   print(wfs.n_snapshots)
   print(wfs.data.shape)

   # TDM
   tdm = read_tdm("static/EigenData_h2p.inp.h5")
   print(tdm.keys())
   print(tdm_of_state("static/EigenData_h2p.inp.h5", 0, axes="x")["tdm_x"].shape)

Visualize
---------

.. code-block:: python

   from zkit.viz import plot_wavefunction, plot_tdm

   plot_wavefunction(
       "td/wfs_h2p.inp.h5",
       step=0,
       outdir="figures",
       npoints=200,
       dpi=150,
       vtk="figures/wfs.vts",
   )

   plot_tdm(
       "static/EigenData_h2p.inp.h5",
       outdir="figures",
       axes=["x", "z"],
       min_mu=1e-3,
       color_by="strength",
       prefix="h2p",
       dpi=150,
   )

Wigner analysis
---------------

.. code-block:: python

   from zkit.mwigner import transform, current, covariance, autocorr, analysis

   # Wigner transform
   W, p = transform.get_wigner_research(psi, dx, dy)

   # Marginal current
   Jx = current.current_x(W, p)

   # Covariance matrix
   Sigma = covariance.compute_4d_covariance(psi, dx, dy)
   nu = covariance.symplectic_eigenvalues(Sigma)
   print(covariance.gaussian_purity(nu))

   # Autocorrelation analysis
   A, A2, P, gap, bound = autocorr.autocorr_analysis(psi_series, dxs)

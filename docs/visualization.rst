Visualization
=============

zkit provides matplotlib-based visualization for wavefunctions and transition
dipole data.

Wavefunction plots
------------------

.. code-block:: python

   from zkit.viz import plot_wavefunction

   path = plot_wavefunction(
       "td/wfs_h2p.inp.h5",
       step=0,
       outdir="figures",
       npoints=240,
       dpi=150,
       axes=None,
       vtk="figures/wfs.vts",
   )
   print(path)

Dispatch by dimension:

- 1D — stacked ``Re/Im ψ`` and ``|ψ|^2``
- 2D — ``|ψ|^2``, ``Re ψ``, and phase panels
- 3D — ``|ψ|^2`` on the ``z = z_mid`` slice

Transition-dipole diagrams
--------------------------

.. code-block:: python

   from zkit.viz import plot_transition_diagram

   plot_transition_diagram(
       "static/EigenData_h2p.inp.h5",
       outfile="figures/tdm_diagram.png",
       axes="x",
       min_mu=1e-3,
       color_by="strength",
       dpi=150,
   )

.. code-block:: python

   from zkit.viz.tdm import plot_tdm_matrix

   plot_tdm_matrix(
       "static/EigenData_h2p.inp.h5",
       outfile="figures/tdm_matrix.png",
       axes=["x", "z"],
       dpi=150,
   )

.. code-block:: python

   from zkit.viz import plot_tdm

   out = plot_tdm(
       "static/EigenData_h2p.inp.h5",
       outdir="figures",
       axes=["x", "z"],
       min_mu=1e-3,
       color_by="strength",
       prefix="h2p",
       dpi=150,
   )

Batch plotting
--------------

.. code-block:: python

   from pathlib import Path
   from zkit.viz import plot_wavefunction

   run_dir = Path("/data/run01")
   td_dir = run_dir / "td"
   out_dir = run_dir / "figures"
   out_dir.mkdir(exist_ok=True)

   for path in sorted(td_dir.glob("wfs_*.h5")):
       plot_wavefunction(path, step=0, outdir=out_dir, npoints=200, dpi=150)

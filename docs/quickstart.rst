Quickstart
==========

This page is the fastest path from a fresh TDSE-Z run to a plotted spectrum,
wavefunction, or transition-dipole diagram.

Prepare a run directory
------------------------

TDSE-Z writes outputs under a single run directory with two subdirectories:

::

    run_dir/
      td/
        <name>_observables.h5
        wfs_<input>.h5
      static/
        <name>_eigen.h5

``<input>`` is the input filename, for example ``h2p.inp``.

Summarize a run from the terminal
---------------------------------

.. code-block:: bash

   zkit summary h2p.inp --run-dir /data/run01

Sample output:

.. code-block:: text

   input=h2p.inp
   dimension=1
   n_steps=480
   time_step=0.050
   final_time=24.000
   evolution: dimension=1 steps=480 dipoles=(480, 4) populations=(480, 4) energies=(480, 7)
   eigen: states=21

Load a run in Python
--------------------

.. code-block:: python

   from zkit.simulation import Run

   run = Run("/data/run01", "h2p.inp")
   print(run)
   print("dimension:", run.dim)
   print("eigenvalues:", run.eigen.values)
   print("evolution:", run.evolution)

Plot a wavefunction snapshot
----------------------------

.. code-block:: bash

   zkit plot-wfs td/h2p_wfs.h5 --step 0 --outdir figures

.. code-block:: python

   from zkit.viz import plot_wavefunction

   path = plot_wavefunction(
       "td/h2p_wfs.h5",
       step=0,
       outdir="figures",
       npoints=240,
       dpi=150,
   )
   print(path)

Plot transition-dipole diagrams
-------------------------------

.. code-block:: bash

   zkit tdm-plot static/h2p_eigen.h5 --axes x,z --outdir figures

.. code-block:: python

   from zkit.viz import plot_tdm

   out = plot_tdm(
       "static/h2p_eigen.h5",
       outdir="figures",
       axes=["x", "z"],
       min_mu=1e-3,
       color_by="strength",
       prefix="h2p",
   )
   print(out["diagram"])
   print(out["matrix"])

Read time-evolution data
------------------------

.. code-block:: python

   from zkit.io.evolution import read_evolution

   evolution = read_evolution("td/h2p_observables.h5")
   print(evolution.time.shape)
   print(evolution.dipoles.shape)
   print(evolution.energies.shape)
   print(evolution.autocorrelation.shape)

Read eigenstates
----------------

.. code-block:: python

   from zkit.io.eigen import read_eigen
   from zkit.io.tdm import tdm_of_state

   eigen = read_eigen("static/h2p_eigen.h5")
   print(eigen.values.shape)
   print(eigen.vectors.shape)

   # transitions out of state 0 along x
   row = tdm_of_state("static/h2p_eigen.h5", 0, axes="x")
   print(row["tdm_x"].shape)

Reconstruct a spatial wavefunction
----------------------------------

.. code-block:: python

   from zkit.io.eigen import read_eigen

   eigen = read_eigen("static/h2p_eigen.h5")
   res = eigen.reconstruct(istate=0, npoints=240)
   print(res["psi"].shape)
   print(res["axes"][0].min(), res["axes"][0].max())

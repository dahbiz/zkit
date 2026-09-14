Realistic post-processing campaign
===================================

This example follows a complete TDSEZ analysis workflow for a 1D
hydrogen-like run. The same pattern applies to 2D and 3D with minor
dimension-dependent adjustments.

Run setup
---------

Assume the TDSEZ binary produced:

::

    /data/h2p/
      td/
        TimeEvolutionData_h2p.inp.h5
        wfs_h2p.inp.h5
      static/
        EigenData_h2p.inp.h5

1. Summarize the run
--------------------

.. code-block:: bash

   zkit summary h2p.inp --run-dir /data/h2p

Confirm the expected dimension, step count, and state count before loading
data.

2. Load everything in Python
----------------------------

.. code-block:: python

   from zkit.simulation import Run

   run = Run("/data/h2p", "h2p.inp")
   print("dimension:", run.dim)
   print("input:", run.meta.input_file)
   print("time step:", run.meta.time_step)
   print("final time:", run.meta.final_time)
   print("n_steps:", run.evolution.n_steps)
   print("states:", run.spectrum.n_states)
   print("wfs snapshots:", run.wavefunctions.n_snapshots)

3. Inspect the spectrum
-----------------------

.. code-block:: python

   import numpy as np

   E = run.spectrum.values
   print("ground state:", E[0])
   print("first excited:", E[1])
   print("spacing:", np.diff(E)[:5])

4. Plot energy over time
------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   time = run.time
   energies = run.energies

   plt.figure(figsize=(8, 4))
   plt.plot(time, energies[:, 4], label="total energy")
   plt.plot(time, energies[:, 1], label="kinetic", alpha=0.6)
   plt.plot(time, energies[:, 2], label="potential", alpha=0.6)
   plt.xlabel("time (a.u.)")
   plt.ylabel("energy (a.u.)")
   plt.legend()
   plt.tight_layout()
   plt.savefig("figures/h2p_energy.png", dpi=150)

5. Plot dipole and acceleration
-------------------------------

.. code-block:: python

   dipoles = run.dipoles
   time = dipoles[:, 0]
   dipole_x = dipoles[:, 2]
   accel_x = dipoles[:, 3]

   plt.figure(figsize=(8, 4))
   plt.plot(time, dipole_x, label="dipole")
   plt.plot(time, accel_x, label="acceleration", alpha=0.7)
   plt.xlabel("time (a.u.)")
   plt.legend()
   plt.tight_layout()
   plt.savefig("figures/h2p_dipole.png", dpi=150)

6. Inspect populations
----------------------

.. code-block:: python

   pop = run.populations
   print(pop.shape)
   print(pop[0, 1:])
   print(pop[-1, 1:])

7. Reconstruct and plot wavefunctions
-------------------------------------

.. code-block:: python

   from zkit.viz import plot_wavefunction

   plot_wavefunction(
       "/data/h2p/td/wfs_h2p.inp.h5",
       step=0,
       outdir="figures",
       npoints=240,
       dpi=150,
       vtk="figures/wfs_step0.vts",
   )

8. Reconstruct selected eigenstates
-----------------------------------

.. code-block:: python

   from zkit.io.eigen import read_eigen

   eigen = read_eigen("/data/h2p/static/EigenData_h2p.inp.h5")
   for i in range(3):
       res = eigen.reconstruct(istate=i, npoints=300)
       x = res["axes"][0]
       psi = res["psi"]
       np.savez(f"figures/eigen_{i}.npz", x=x, psi=psi)

9. Check norm conservation
--------------------------

.. code-block:: python

   from zkit.io.wfs_field import compute_wfs_norm
   from pathlib import Path

   wfs_path = Path("/data/h2p/td/wfs_h2p.inp.h5")
   for step in [0, 50, 100, 150]:
       print(step, compute_wfs_norm(wfs_path, step=step, nq=5))

10. Export selected snapshots to VTK for ParaView
--------------------------------------------------

.. code-block:: python

   from zkit.io.wfs_field import reconstruct_wfs, wfs_to_vtk
   from pathlib import Path

   outdir = Path("figures/vtk")
   outdir.mkdir(exist_ok=True)

   for step in [0, 50, 100]:
       res = reconstruct_wfs(wfs_path, step=step, npoints=120)
       wfs_to_vtk(res, outdir / f"wfs_step{step}.vts")

Resulting artifacts
-------------------

- ``figures/h2p_energy.png``
- ``figures/h2p_dipole.png``
- ``figures/wfs_step0.png``
- ``figures/wfs_step0.vts``
- ``figures/eigen_0.npz``, ``eigen_1.npz``, ``eigen_2.npz``
- ``figures/vtk/wfs_step{0,50,100}.vts``

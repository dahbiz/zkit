Transition-dipole campaign
===========================

This example computes and inspects transition-dipole matrices for a small
eigenbasis, then extracts useful subsets for spectroscopy.

Data
----

::

    /scratch/znse/
      static/
        EigenData_znse_d20.h5
        TDM_Dx_znse_d20.npy
        EigenEnergies_znse_d20.npy
        States_znse_d20.npy

1. Read the spectrum
--------------------

.. code-block:: python

   from zkit.io.eigen import read_eigen

   eigen = read_eigen("/scratch/znse/static/EigenData_znse_d20.h5")
   E = eigen.values
   print("states:", E.shape[0])
   print("lowest 5 energies:", E[:5])

2. Read the full TDM
--------------------

.. code-block:: python

   from zkit.io.tdm import read_tdm, tdm_of_state, tdm_magnitude_of_state

   tdm = read_tdm("/scratch/znse/static/EigenData_znse_d20.h5")
   print("axes present:", list(tdm.keys()))

   Dx = tdm["tdm_x"]
   print("Dx shape:", Dx.shape)
   print("Hermitian check:", np.allclose(Dx, Dx.T.conj()))

3. Extract strongest couplings from ground state
------------------------------------------------

.. code-block:: python

   import numpy as np

   row0 = tdm_of_state("/scratch/znse/static/EigenData_znse_d20.h5", 0, axes="x")
   mag = np.abs(row0["tdm_x"])

   idx = np.argsort(mag)[::-1]
   for j in idx[:10]:
       print(j, E[j] - E[0], mag[j])

4. Plot a filtered transition diagram
-------------------------------------

.. code-block:: python

   from zkit.viz import plot_transition_diagram

   plot_transition_diagram(
       "/scratch/znse/static/EigenData_znse_d20.h5",
       outfile="figures/znse_diagram_x.png",
       axes="x",
       min_mu=0.05,
       color_by="strength",
       dpi=200,
   )

5. Plot a heatmap
-----------------

.. code-block:: python

   from zkit.viz.tdm import plot_tdm_matrix

   plot_tdm_matrix(
       "/scratch/znse/static/EigenData_znse_d20.h5",
       outfile="figures/znse_matrix_xz.png",
       axes=["x", "z"],
       dpi=200,
   )

6. Oscillator strength summary
------------------------------

.. code-block:: python

   from zkit.viz.tdm import _oscillator_strength

   # combined magnitude over requested axes
   mu = np.zeros((E.shape[0], E.shape[0]))
   for m in tdm.values():
       mu += np.abs(m) ** 2
   mu = np.sqrt(mu)

   active_dim = len(tdm)
   fmat = _oscillator_strength(E, mu, active_dim)
   print("strongest f:", np.max(fmat))
   print("strongest pair:", np.unravel_index(np.argmax(fmat), fmat.shape))

7. Save selected tables
-----------------------

.. code-block:: python

   import numpy as np

   # strongest 200 transitions from ground state
   row0 = tdm_of_state("/scratch/znse/static/EigenData_znse_d20.h5", 0)
   mag = np.sqrt(sum(np.abs(v) ** 2 for v in row0.values()))
   order = np.argsort(mag)[::-1][:200]

   selected = {
       "from": np.zeros(200, dtype=np.int32),
       "to": order.astype(np.int32),
       "dE": E[order] - E[0],
       "mu": mag[order],
   }
   np.savez("figures/znse_ground_transitions.npz", **selected)

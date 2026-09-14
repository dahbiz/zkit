Transition dipole moments
=========================

TDSEZ computes dipole couplings between eigenstates. zkit reads the static
eigenstate output and exposes both full matrices and per-state rows.

Read full TDM matrices
----------------------

.. code-block:: python

   from zkit.io.tdm import read_tdm

   tdm = read_tdm("static/EigenData_h2p.inp.h5")
   print(tdm.keys())
   print(tdm["tdm_x"].shape)
   print(tdm["tdm_x"].dtype)

If the file predates TDM output or no dipole axis was assembled, an empty
dict is returned.

Extract transitions out of one state
------------------------------------

.. code-block:: python

   from zkit.io.tdm import tdm_of_state, tdm_magnitude_of_state

   row = tdm_of_state("static/EigenData_h2p.inp.h5", 0, axes="x")
   print(row["tdm_x"].shape)

   mag = tdm_magnitude_of_state("static/EigenData_h2p.inp.h5", 0, axes=["x", "z"])
   print(mag.shape)

Plot diagrams and heatmaps
--------------------------

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
   print(out["diagram"])
   print(out["matrix"])

This writes:

- ``figures/h2p_diagram.png``
- ``figures/h2p_matrix.png``

Oscillator strength coloring
----------------------------

.. code-block:: python

   from zkit.viz.tdm import plot_transition_diagram

   plot_transition_diagram(
       "static/EigenData_h2p.inp.h5",
       outfile="figures/h2p_f_diagram.png",
       axes="x",
       min_mu=1e-3,
       color_by="f",
       dpi=150,
   )

``color_by="f"`` colors arrows by oscillator strength

.. math::

   f_{ij} = \frac{2}{d} (E_j - E_i) |\mu_{ij}|^2.

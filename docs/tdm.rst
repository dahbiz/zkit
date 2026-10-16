Transition dipole moments
=========================

The transition dipoles :math:`d^{(a)}_{ij} = \langle i|x_a|j\rangle` between the
eigenstates of a run are computed by TDSE-Z's post-processing tool
``dipoles.x``, in 1D, 2D and 3D. The run must save the eigenstates and the
dipole operators; then run the tool in the run directory:

.. code-block:: text

   # in the input file
   NBoundStatesSave = 1
   SaveDipoleAxes   = xyz        # the axes of the problem

   $ ./dipoles.x h2p.inp          # writes static/h2p_tdm_Dx.npy, _Dy, _Dz, ...

zkit reads these files next to ``static/<name>_eigen.h5`` (also under the
names of earlier versions) and exposes the full matrices and per-state rows.

Read full TDM matrices
----------------------

.. code-block:: python

   from zkit.io.tdm import read_tdm

   tdm = read_tdm("static/h2p_eigen.h5")
   print(tdm.keys())
   print(tdm["tdm_x"].shape)
   print(tdm["tdm_x"].dtype)

If ``dipoles.x`` has not been run, an empty dict is returned.

Check against exact results
---------------------------

:mod:`zkit.analytic` gives the exact spectrum and dipoles of an oscillator
(1D, 2D, 3D, one frequency per axis) and of a hard-wall box, and
:func:`~zkit.analytic.compare_tdm` matches the numerical states to them by
energy and compares :math:`|d_{ij}|` (eigenvector phases are arbitrary):

.. code-block:: python

   import zkit
   from zkit.analytic import compare_tdm, oscillator_tdm
   from zkit.io.tdm import read_tdm

   run = zkit.Run(".", "ho3.inp")      # V = (0.36 x^2 + 0.6889 y^2 + 1.3689 z^2)/2
   ref = oscillator_tdm((0.6, 0.83, 1.17), n_states=6)
   print(compare_tdm(read_tdm(run.path("eigen")), run.spectrum.values, ref))
   # {'energies': ..., 'x': ..., 'y': ..., 'z': ...}   relative errors

``zkit validate`` runs these comparisons for TDSE-Z (:doc:`validation`).

Extract transitions out of one state
------------------------------------

.. code-block:: python

   from zkit.io.tdm import tdm_of_state, tdm_magnitude_of_state

   row = tdm_of_state("static/h2p_eigen.h5", 0, axes="x")
   print(row["tdm_x"].shape)

   mag = tdm_magnitude_of_state("static/h2p_eigen.h5", 0, axes=["x", "z"])
   print(mag.shape)

Plot diagrams and heatmaps
--------------------------

.. code-block:: python

   from zkit.viz import plot_tdm

   out = plot_tdm(
       "static/h2p_eigen.h5",
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
       "static/h2p_eigen.h5",
       outfile="figures/h2p_f_diagram.png",
       axes="x",
       min_mu=1e-3,
       color_by="f",
       dpi=150,
   )

``color_by="f"`` colors arrows by oscillator strength

.. math::

   f_{ij} = \frac{2}{d} (E_j - E_i) |\mu_{ij}|^2.

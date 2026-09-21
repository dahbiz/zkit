Troubleshooting
===============

``import zkit.physics`` fails with ``ModuleNotFoundError: No module named 'sympy'``
------------------------------------------------------------------------------------

Install sympy:

.. code-block:: bash

   pip install sympy

``zkit.physics`` is deck-parsing code that depends on sympy for symbolic
differentiation of ``Potential``, ``Mass``, and laser expressions.

``import zkit.io.wfs_field`` or ``eigen.reconstruct()`` fails
--------------------------------------------------------------

The evaluator is bundled with zkit as ``igakit.igalib.bsp``.  Reinstall zkit
if the module is missing:

.. code-block:: bash

   pip install -e .

``plot_wavefunction(..., vtk=...)`` fails
-----------------------------------------

Install pyvista:

.. code-block:: bash

   pip install pyvista

VTK export is not declared in ``pyproject.toml`` core deps.

``zkit.mwigner`` fails
-----------------------

Install the optional extra:

.. code-block:: bash

   pip install -e ".[mwigner]"

This installs ``torch``.

Wavefunction reconstruction raises about missing ``knots_*``
------------------------------------------------------------

Re-run TDSEZ with the current binary. Older flat ``wfs_*.h5`` files do not
embed the knot vectors and are not supported by the current reconstruction
path.

HDF5 file not found or empty
-----------------------------

Check that the run directory contains ``td/`` and ``static/`` subdirectories and
that the basename passed to ``Run(run_dir, basename)`` matches the files:

::

    run_dir/
      td/TimeEvolutionData_<basename>.h5
      static/EigenData_<basename>.h5

Slow norm or reconstruction for large grids
--------------------------------------------

Reduce ``npoints`` or ``nq``. For production reconstructions on large datasets,
stream outputs instead of holding full grids in memory.

PETSc MPI slowdown
------------------

If running through MPI wrappers, pinning can serialize execution on one core.
Use ``--bind-to none`` or ``--oversubscribe`` if necessary.

Wigner transform memory
-----------------------

Large 2D arrays can exceed RAM. Process snapshots one at a time or downsample
the grid.

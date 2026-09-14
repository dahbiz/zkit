Command-line interface
======================

The ``zkit`` command-line tool provides quick summaries, plotting, and batch
operations without writing Python code.

Global option
-------------

::

    zkit [--run-dir DIR] <subcommand> ...

``--run-dir`` defaults to ``.``.

Subcommands
-----------

summary
~~~~~~~

Summarize one simulation run.

.. code-block:: bash

   zkit summary h2p.inp --run-dir /data/run01

Shows dimension, step count, time step, final time, evolution dataset shapes,
and eigenstate count if ``static/EigenData_<input>.h5`` is present.

batch
~~~~~

Summarize every ``TimeEvolutionData_*.h5`` under ``run_dir/td/``.

.. code-block:: bash

   zkit batch --run-dir /data/run01

Outputs CSV to stdout:

.. code-block:: text

   basename,dimension,steps,final_time
   h2p,1,480,24.0
   he_1d,1,320,16.0

plot-wfs
~~~~~~~~

Render one wavefunction snapshot to PNG.

.. code-block:: bash

   zkit plot-wfs td/wfs_h2p.inp.h5 \
     --step 0 \
     --outdir figures \
     --npoints 240 \
     --dpi 150 \
     --vtk figures/wfs.vts \
     --norm

Options:

- ``step`` snapshot index into the leading time axis.
- ``outdir`` output directory; created if missing.
- ``npoints`` grid points per spatial axis.
- ``dpi`` figure DPI.
- ``vtk`` optional VTK export path; requires ``pyvista``.
- ``norm`` print the exact ``⟨ψ|ψ⟩`` via Gauss quadrature over knot spans.

tdm-plot
~~~~~~~~

Plot transition-dipole diagrams and the ``|μ_ij|`` heatmap from an
``EigenData_*.h5`` file.

.. code-block:: bash

   zkit tdm-plot static/EigenData_h2p.inp.h5 \
     --run-dir /data/run01 \
     --axes x,z \
     --min-mu 1e-3 \
     --color-by strength \
     --outdir figures \
     --prefix h2p \
     --dpi 150

Options:

- ``eig_file`` EigenData path, or just the basename when combined with ``--run-dir``.
- ``--axes`` comma-separated axes, for example ``x,y,z``.
- ``--min-mu`` minimum transition magnitude to draw.
- ``--color-by`` ``strength`` or ``oscillator strength``.
- ``--outdir`` output directory.
- ``--prefix`` output filename prefix.
- ``--dpi`` figure DPI.

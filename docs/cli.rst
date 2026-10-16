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
and eigenstate count if ``static/<name>_eigen.h5`` is present.

batch
~~~~~

Summarize every ``<name>_observables.h5`` under ``run_dir/td/``.

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

   zkit plot-wfs td/h2p_wfs.h5 \
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
``<name>_eigen.h5`` file.

.. code-block:: bash

   zkit tdm-plot static/h2p_eigen.h5 \
     --run-dir /data/run01 \
     --axes x,z \
     --min-mu 1e-3 \
     --color-by strength \
     --outdir figures \
     --prefix h2p \
     --dpi 150

Options:

- ``eig_file`` ``static/<name>_eigen.h5`` path, or the input name when combined with ``--run-dir``.
- ``--axes`` comma-separated axes, for example ``x,y,z``.
- ``--min-mu`` minimum transition magnitude to draw.
- ``--color-by`` ``strength`` or ``oscillator strength``.
- ``--outdir`` output directory.
- ``--prefix`` output filename prefix.
- ``--dpi`` figure DPI.

pes
~~~

Photoelectron energy spectrum dP/dE from a grid-mode t-SURFF file
(``td/<name>_surff.h5``), optionally against a SCID-TDSE ``sts_volkov_table``
(plain or gzipped). Writes ``<prefix>_dPdE.txt`` and ``<prefix>_dPdE.png``.

.. code-block:: bash

   zkit pes td/quick_surff.h5 --scid scid/pes_volkov.table.gz --outdir figures

Without ``--scid`` the spectrum is evaluated on ``--nk`` ``|k|`` points up to the
grid edge and ``--ntheta`` polar angles.

compare-scid
~~~~~~~~~~~~

Compare a TDSE-Z run with a SCID-TDSE run one to one: field, dipole and
acceleration, harmonic spectrum, dP/dE (yields per energy window), angular
distributions at the ATI peaks and momentum maps. See :doc:`photoelectron`.

.. code-block:: bash

   zkit --run-dir tdsez compare-scid quick.inp \
     --scid-detail scid/detail.table.gz --scid-pes scid/pes_volkov.table.gz \
     --omega 0.4 --e0 0.05 --regions 0:0.12,0.12:0.47,0.47:0.9,0.9:1.2 \
     --outdir figures

Prints and writes ``<prefix>_metrics.txt`` and the figures
``<prefix>_{time,harmonics,pes,angular,kmap}.png``.

validate
~~~~~~~~

Run TDSE-Z on analytic cases and check every output column; see
:doc:`validation`.

.. code-block:: bash

   zkit validate --binary /path/to/tdsez.x --np 8 [--reference old/tdsez.x] [--quick]

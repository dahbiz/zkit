Quick 1D and 2D oscillator runs
================================

The repository includes two small TDSEZ input decks.  They solve three states
of a one-dimensional harmonic oscillator and six states of an isotropic
two-dimensional oscillator.  Propagation is disabled, so each run only does a
static eigenvalue solve and normally completes in a few seconds including
MPI/PETSc startup.

Build TDSEZ in a separate directory, then run the examples from the zkit
checkout::

   python -m pip install -e ".[viz]"
   cmake -S /path/to/tdsez -B /tmp/tdsez-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
   cmake --build /tmp/tdsez-build --target tdsez --parallel 4
   python examples/run_quick_examples.py --tdsez /tmp/tdsez-build/tdsez

The runner writes the copied decks, HDF5 eigenvalue files, CSV spectra, and
PNG plots below ``examples/output``.  Use ``--case 1d`` or ``--case 2d`` to
run only one dimension.  The expected energies are ``[0.1, 0.3, 0.5]`` in 1D
and ``[0.2, 0.4, 0.4, 0.6, 0.6, 0.6]`` in 2D (atomic units).

Reference spectra from this run:

.. image:: ../_static/examples/quick-ho1d-spectrum.png
   :alt: 1D harmonic oscillator spectrum
   :width: 600px

.. image:: ../_static/examples/quick-ho2d-spectrum.png
   :alt: 2D harmonic oscillator spectrum
   :width: 600px

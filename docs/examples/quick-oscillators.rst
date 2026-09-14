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

The runner also includes a finite quantum-well heterostructure:

.. code-block:: bash

   python examples/run_quick_examples.py --tdsez /tmp/tdsez-build/tdsez --case heterostructure

This example uses a 0.6 a.u. conduction-band offset outside ``-4 < x < 4``
and a 20% larger barrier effective mass.  It produces ``profile.csv`` and a
``potential_and_levels.png`` plot showing the band profile, effective mass,
and computed bound-state energies.

.. image:: ../_static/examples/heterostructure-potential-levels.png
   :alt: Heterostructure potential, effective mass, and bound-state levels
   :width: 650px

Driven dynamics and Rabi-style transfer
---------------------------------------

The ``rabi`` case starts in the harmonic-oscillator ground state and applies
``E_x(t) = 0.05 sin(0.2 t)``.  The carrier is resonant with the 0-to-1 level
spacing.  Run it with::

   python examples/run_quick_examples.py --tdsez /tmp/tdsez-build/tdsez --case rabi

The resulting ``rabi-populations-response.png`` contains bound-state
populations, the applied electric field, and the dipole response.  The CSV
file contains those observables for further Fourier, susceptibility, or
quantum-beating analysis.

.. image:: ../_static/examples/ho1d-rabi-populations-response.png
   :alt: Rabi-style bound-state populations, laser field, and dipole response
   :width: 650px

The runner also creates ``energy-current-spectra.png``.  It combines the
kinetic, potential, interaction, and total energies; the intra-band,
inter-band, and total currents; and normalized Fourier spectra of the dipole
and autocorrelation.  These are useful diagnostics for energy exchange,
selection rules, and quantum beating.

.. image:: ../_static/examples/ho1d-energy-current-spectra.png
   :alt: Energy decomposition, current decomposition, and response spectra
   :width: 650px

Reference spectra from this run:

.. image:: ../_static/examples/quick-ho1d-spectrum.png
   :alt: 1D harmonic oscillator spectrum
   :width: 600px

.. image:: ../_static/examples/quick-ho2d-spectrum.png
   :alt: 2D harmonic oscillator spectrum
   :width: 600px

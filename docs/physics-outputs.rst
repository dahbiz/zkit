Physics outputs
===============

TDSEZ's time monitor can emit dipole, energy, population, current, and
autocorrelation observables. zkit reads them directly from the time-evolution
HDF5 file.

Available observables
---------------------

.. list-table:: Available observables
   :header-rows: 1

   * - Field
     - Columns
     - Description
   * - dipoles
     - t
     - time
   * - dipoles
     - Ex/Ey/Ez
     - laser envelope
   * - dipoles
     - Dx/Dy/Dz
     - dipole projection
   * - dipoles
     - Ax/Ay/Az
     - acceleration
   * - populations
     - t, P0..
     - populations
   * - energies
     - t, kinetic, potential, interaction, total, inv_mass_avg, norm_sq
     - energy decomposition
   * - currents
     - 11 columns in 1D/2D, 15 in 3D
     - current decomposition
   * - autocorrelation
     - t, Re A(t), Im A(t)
     - autocorrelation function

Read observables
----------------

.. code-block:: python

   from zkit.io.evolution import read_evolution

   evolution = read_evolution("td/TimeEvolutionData_h2p.inp.h5")

   time = evolution.time
   dipoles = evolution.dipoles
   populations = evolution.populations
   energies = evolution.energies
   currents = evolution.currents
   autocorrelation = evolution.autocorrelation

   print(time.shape, dipoles.shape, populations.shape)
   print(energies.shape, currents.shape, autocorrelation.shape)

Column names
------------

.. code-block:: python

   from zkit.layouts import dipole_layout, AC_COLUMNS, ENERGY_COLUMNS, current_columns

   layout = dipole_layout(1)
   print(layout.asdict())
   print(ENERGY_COLUMNS)
   print(AC_COLUMNS)
   print(current_columns(1))
   print(current_columns(3))

Physics from input deck
-----------------------

.. code-block:: python

   from zkit.simulation import Run

   run = Run("run_dir", "h2p.inp")
   phys = run.physics

   print(phys.dim)
   print(phys.t_step)
   print(phys.stride_wfs)
   print(phys.has_laser)
   print(phys.snapshot_time(10))

   V = phys.V(0.0, 0.0)
   Ax = phys.Ax(1000.0)
   Vpx = phys.Vpx(0.1, 0.0)

Deck-derived physics requires ``sympy``:

.. code-block:: bash

   pip install sympy

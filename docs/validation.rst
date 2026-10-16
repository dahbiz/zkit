Physics validation
==================

``zkit validate`` runs TDSE-Z on cases with exact solutions or exact
identities, reads every output table through :class:`zkit.Run`, and checks
every column. With a second binary it also compares every column of every
table between the two builds, so a change to TDSE-Z can be shown to leave the
physics unchanged.

.. code-block:: bash

   pip install -e ".[validation]"      # from a clone of the zkit repository
   export TDSEZ_BINARY=/path/to/tdsez.x
   zkit validate --quick                      # 2D cases, about 20 s
   zkit validate --np 8                       # all cases, about 10 min
   zkit validate --np 8 --reference old/tdsez.x

The command exits with 0 when every check passes. Runs are kept in
``--workdir`` (default ``zkit-validation/``) for inspection.

Conventions
-----------

``q = Charge`` (``-1`` electron), ``s = -q``: the carrier sees ``s V`` and the
field term ``s E.r``. Columns of ``td/<name>_observables.h5``:

.. code-block:: text

   dipoles      2D: t Ex Ey Dx Dy Ax Ay      3D: t Ex Ey Ez Dx Dy Dz Ax Ay Az
   energies     t Kin Pot Int Tot InvMass Norm
   currents     t Lz dGamma J.. J_intra.. J_inter.. J_bc..
   populations  t P_0 ...
   autocorrelation  t Re Im of <psi(0)|M|psi(t)>

Cases
-----

**Transition dipoles** (1D, 2D, 3D). A stationary run saves the eigenstates
and the dipole operators; ``dipoles.x`` computes every :math:`d^{(a)}_{ij}`.
The spectrum and every :math:`|d_{ij}|` are compared with
:mod:`zkit.analytic`: oscillators with one frequency per axis
(:math:`\omega = 0.5`; :math:`(0.6, 0.83)`; :math:`(0.6, 0.83, 1.17)`),
:math:`\langle n|x_a|n+1\rangle = \sqrt{(n+1)/2\omega_a}`, and the hard-wall
box of width 10, :math:`\langle n|x|m\rangle = -8Lnm/(\pi^2(n^2-m^2)^2)` for
:math:`n+m` odd. The rows of ``dipole_spectrum.x`` and ``recombination.x``
must equal the full matrix on every axis. ``--quick`` skips the 3D case.

**Static Kramers-Henneberger frame** (``Mode = kh``; 1D linear with
:math:`m = 2`, 2D circular, 3D linear). For an oscillator the cycle-averaged
potential is :math:`V + \tfrac12 m\omega_0^2\langle|\boldsymbol\alpha|^2\rangle`,
so every level shifts by :math:`\tfrac14 m\omega_0^2\alpha_0^2` (linear) or
:math:`\tfrac12 m\omega_0^2\alpha_0^2` (circular),
:math:`\alpha_0 = |q|E_0/m\omega^2` (:func:`zkit.analytic.kh_oscillator_shift`).
The KH levels minus the ``Mode = td`` levels on the same grid must equal the
shift (the discretisation cancels), and the KH levels the closed form.

**Free Gaussian packet in a uniform field** (3D, mass 0.7, ``q = -1`` and
``+1``). For a linear potential Ehrenfest is exact:

.. math::

   \langle x_a\rangle = x_{0,a} + \frac{k_{0,a}t}{m} - \frac{s}{m}\int_0^t\!\!\int_0^\tau E_a,
   \qquad \langle v_a\rangle = \frac{k_{0,a} - s\int_0^t E_a}{m},
   \qquad A_a = -\frac{q s E_a}{m},

the kinetic energy is
:math:`\sum_a [m\langle v_a\rangle^2/2 + 1/(8 m w_a^2)]`, and the
autocorrelation follows by quadrature in momentum space (gauge transformation
to the velocity gauge). The field integrals are taken symbolically from the
deck's expressions.

**Driven harmonic oscillator** (2D, mass 1.5, ``q = -1`` and ``+1``), started
in the ground state: the state stays a coherent state
:math:`|\alpha(t)\rangle e^{i\varphi}` with
:math:`\dot\alpha = -i\omega\alpha - ig`,
:math:`\dot\varphi = -\omega/2 - g\,\mathrm{Re}\,\alpha`,
:math:`g = sE/\sqrt{2m\omega}`. This fixes ``D``, ``J``, ``A``, ``Kin``,
``Pot``, ``Int``, ``Tot``, ``P_0 = |<0|psi>|^2``, the complex autocorrelation,
the accumulated Berry phase and ``E_0 = omega``.

**Position-dependent mass** (2D and 3D; no closed form). A field-free
eigenstate is stationary: ``D`` constant, ``J = A = 0``, ``Tot = E_0``,
``P_0 = 1`` and autocorrelation :math:`e^{-iE_0t}`. A driven state obeys
Ehrenfest, :math:`J = \dot D` and :math:`A = \ddot D` (fourth-order
differences), and the operator and discrete accelerations agree. Both forms
are tested.

**Every run** additionally checks the carrier sign and the field columns
against :class:`zkit.physics.Physics`, the norm, and the energy theorem for a
Hermitian Hamiltonian,

.. math::

   \mathrm{Tot}(t) - \mathrm{Tot}(0) = \int_0^t s\,\dot{\mathbf E}\cdot\langle\mathbf r\rangle\,d\tau ,

with :math:`\dot{\mathbf E}` differentiated symbolically
(:meth:`zkit.physics.Physics.field_derivative`).

Tolerances and what limits them
-------------------------------

Exact identities (field columns, ``<1/m>``, the current decomposition, norms,
stationary states with the discrete acceleration) are checked to rounding.
Comparisons with exact dynamics are limited by the spatial discretisation and
the time step; on the validation grids they agree to 3e-5 or better. Derived
checks that differentiate or integrate the output in time (``d2D/dt2``, the
energy theorem) inherit that error amplified, and their tolerances say so. The
operator-form acceleration of an eigenstate is zero only to the spatial
discretisation error, which falls as :math:`h^4` (degree 4): 5.2e-6 at
:math:`h = 1.1`, 8.2e-7 at :math:`h = 0.75` in 3D.

The free packet is stopped at ``t = 3``, before its tail reaches the
reflecting walls of the box.

API
---

.. automodule:: zkit.validation
   :members:

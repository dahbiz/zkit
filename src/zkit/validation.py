"""Physics validation of TDSE-Z against analytic results (``zkit validate``).

Runs TDSE-Z on cases with exact solutions or exact identities, reads every
output table through :class:`zkit.Run` and checks every column:

* **free Gaussian packet in a uniform field** (3D, mass 0.7, both charge
  signs): Ehrenfest is exact for a linear potential, so <r>, <v>, the
  acceleration, the kinetic energy and the autocorrelation (momentum-space
  quadrature) are known exactly;
* **driven harmonic oscillator from its ground state** (2D, mass 1.5, both
  signs): the state stays a coherent state, which gives <r>, <p>, all
  energies, the ground-state population, the complex autocorrelation and the
  Berry phase;
* **transition dipoles** (1D, 2D, 3D): the spectrum and every |<i|x_a|j>| of
  the post-processor ``dipoles.x`` against the exact oscillator (distinct
  frequencies per axis) and hard-wall box results (:mod:`zkit.analytic`), and
  the rows of ``dipole_spectrum.x`` and ``recombination.x`` against the full matrix;
* **static Kramers-Henneberger frame** (``Mode = kh``; 1D, 2D circular, 3D):
  oscillator levels shift by m w0^2 <|alpha|^2>/2 exactly, against a lab-frame
  run on the same grid and against the closed-form levels;
* **position-dependent mass** (2D, 3D; no closed form): a field-free
  eigenstate is stationary (D, J, A, energies constant, autocorrelation
  exp(-i E0 t)); a driven state obeys Ehrenfest (J = dD/dt, A = d2D/dt2), the
  energy theorem dTot/dt = s E'(t).<r>, and the operator and discrete
  accelerations agree.

With a reference binary every column of every table is also compared between
the two builds.

TDSE-Z conventions: q = Charge, s = -q; the carrier sees s V and the field
term s E.r. Tables of ``td/<name>_observables.h5``::

    dipoles   2D: t Ex Ey Dx Dy Ax Ay      3D: t Ex Ey Ez Dx Dy Dz Ax Ay Az
    energies  t Kin Pot Int Tot InvMass Norm
    currents  t Lz dGamma J.. J_intra.. J_inter.. J_bc..
    populations t P_0 ...     autocorrelation t Re Im of <psi(0)|M|psi(t)>
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

TABLES = ("dipoles", "energies", "currents", "populations", "autocorrelation")


@dataclass
class Check:
    case: str
    quantity: str
    error: float
    tol: float

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.error) and self.error <= self.tol)


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    diffs: Dict[str, float] = field(default_factory=dict)  # "case/table" -> max column difference

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks) and all(d <= 1e-10 for d in self.diffs.values())

    def text(self) -> str:
        lines = [f"{'case':42s} {'quantity':34s} {'error':>9s} {'tol':>7s}"]
        for c in self.checks:
            lines.append(
                f"{c.case:42s} {c.quantity:34s} {c.error:9.1e} {c.tol:7.0e}  {'ok' if c.ok else 'FAIL'}"
            )
        lines.append(f"\n{sum(c.ok for c in self.checks)}/{len(self.checks)} checks passed")
        if self.diffs:
            lines.append(
                "\ncurrent vs reference binary, max |new - ref| / max(|ref|, 1) over all columns:"
            )
            for k, v in sorted(self.diffs.items()):
                lines.append(f"  {k:40s} {v:9.1e}  {'ok' if v <= 1e-10 else 'DIFFERS'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analytic references (no TDSE-Z needed; unit tested)
# ---------------------------------------------------------------------------


def time_derivative(y: np.ndarray, h: float, order: int = 1) -> np.ndarray:
    """Fourth-order central difference on interior points [2:-2] (axis 0)."""
    if order == 1:
        return (-y[4:] + 8 * y[3:-1] - 8 * y[1:-3] + y[:-4]) / (12 * h)
    return (-y[4:] + 16 * y[3:-1] - 30 * y[2:-2] + 16 * y[1:-3] - y[:-4]) / (12 * h * h)


def free_packet(t, m, k0, x0, s, int_E, int2_E):
    """Free carrier in a uniform field, H = p^2/2m + s E(t).r (exact Ehrenfest).

    ``int_E[a](t)`` = int_0^t E_a, ``int2_E[a](t)`` = int_0^t int_0^tau E_a.
    Returns <r>(t) and <v>(t), shape (nt, d)."""
    t = np.asarray(t, float)
    r = np.stack([x0[a] + k0[a] * t / m - (s / m) * int2_E[a](t) for a in range(len(k0))], 1)
    v = np.stack([(k0[a] - s * int_E[a](t)) / m for a in range(len(k0))], 1)
    return r, v


def free_packet_autocorrelation(t, m, w, k0, x0, s, int_E, n_k=6001, n_tau=200001):
    """<psi0|psi(t)> for the Gaussian packet psi0 = prod_a (2 pi w^2)^(-1/4)
    exp(-(x-c)^2/4w^2 + i k0 x) under H = p^2/2m + s E(t).r, by quadrature in
    momentum space (gauge psi = e^{iA.r} chi, A' = -s E):

        prod_a int conj(phi0(k)) phi0(k - A(t)) exp(-i int_0^t (k - A(t) + A(tau))^2 / 2m dtau) dk.
    """
    from scipy.integrate import cumulative_trapezoid, trapezoid

    t = np.asarray(t, float)
    taus = np.linspace(0.0, t[-1], n_tau)
    ac = np.ones(len(t), complex)
    for a in range(len(k0)):
        A = -s * int_E[a](taus)
        iA = cumulative_trapezoid(A, taus, initial=0)
        iA2 = cumulative_trapezoid(A * A, taus, initial=0)
        k = np.linspace(k0[a] - 12 / w[a], k0[a] + 12 / w[a], n_k)

        def phi0(kk, a=a):
            return (2 * w[a] ** 2 / np.pi) ** 0.25 * np.exp(
                -(w[a] ** 2) * (kk - k0[a]) ** 2 - 1j * (kk - k0[a]) * x0[a]
            )

        for i, tt in enumerate(t):
            j = int(round(tt / taus[1])) if len(taus) > 1 and taus[1] > 0 else 0
            kk = k - A[j]
            phase = (kk * kk * tt + 2 * kk * iA[j] + iA2[j]) / (2 * m)
            ac[i] *= trapezoid(np.conj(phi0(k)) * phi0(k - A[j]) * np.exp(-1j * phase), k)
    return ac


def coherent_state(t, m, omega, s, fields: Sequence[Callable]):
    """Driven oscillator H = sum_a [p_a^2/2m + m w^2 x_a^2/2 + s E_a(t) x_a] from
    the ground state: |alpha(t)> e^{i phi}, alpha' = -i w alpha - i g,
    phi' = -w/2 - g Re(alpha), g = s E / sqrt(2 m w).

    Returns <x>, <p> (nt, d) and the amplitude <0|psi(t)> (product over axes)."""
    from scipy.integrate import solve_ivp

    t = np.asarray(t, float)
    X = np.zeros((len(t), len(fields)))
    P = np.zeros_like(X)
    amp = np.ones(len(t), complex)
    for a, E in enumerate(fields):

        def rhs(tt, y, E=E):
            g = s * E(tt) / np.sqrt(2 * m * omega)
            return [omega * y[1], -omega * y[0] - g, -omega / 2 - g * y[0]]

        sol = solve_ivp(
            rhs, (0.0, t[-1]), [0.0, 0.0, 0.0], t_eval=t, rtol=1e-12, atol=1e-14, method="DOP853"
        )
        ar, ai, ph = sol.y
        X[:, a] = np.sqrt(2 / (m * omega)) * ar
        P[:, a] = np.sqrt(2 * m * omega) * ai
        amp *= np.exp(1j * ph - (ar * ar + ai * ai) / 2)
    return X, P, amp


def field_work(t, r, s, dE: Sequence[Callable]):
    """int_0^t s E'(tau).<r>(tau) dtau (Simpson), the change of the total energy
    <H> + s E.<r> for a Hermitian Hamiltonian."""
    from scipy.integrate import cumulative_simpson

    t = np.asarray(t, float)
    integrand = np.sum(s * np.stack([f(t) for f in dE], 1) * r, 1)
    return cumulative_simpson(integrand, x=t, initial=0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def default_binary() -> Optional[str]:
    """TDSEZ_BINARY, then tdsez.x / tdsez on PATH."""
    b = os.environ.get("TDSEZ_BINARY")
    if b:
        return b
    for name in ("tdsez.x", "tdsez"):
        p = shutil.which(name)
        if p:
            return p
    return None


class Validator:
    """Run the validation cases with ``binary`` (and ``reference``) in ``workdir``."""

    def __init__(
        self, binary, reference=None, nproc=4, workdir="zkit-validation", mpiexec="mpiexec"
    ):
        self.binary, self.reference, self.nproc = binary, reference, int(nproc)
        self.workdir = Path(workdir)
        self.mpiexec = mpiexec
        self.report = Report()

    # -- execution ---------------------------------------------------------
    def _run(self, name: str, text: str, binary: str):
        from zkit.simulation import Run

        d = self.workdir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "in.inp").write_text(text)
        cmd = [self.mpiexec, "-n", str(self.nproc)]
        if "mpi" in os.path.basename(self.mpiexec):
            cmd += ["--bind-to", "none"]
        r = subprocess.run(cmd + [binary, "-inp", "in.inp"], cwd=d, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{name}: TDSE-Z failed\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return Run(d, "in.inp")

    def run(self, name: str, text: str):
        run = self._run(name, text, self.binary)
        if self.reference:
            ref = self._run(name + "_ref", text, self.reference)
            for tab in TABLES:
                a, b = getattr(run, tab), getattr(ref, tab)
                if a.shape != b.shape:
                    self.report.diffs[f"{name}/{tab}"] = float("inf")
                    continue
                scale = np.maximum(np.max(np.abs(b), axis=0), 1.0)
                self.report.diffs[f"{name}/{tab}"] = float(np.max(np.abs(a - b) / scale))
        return run

    # -- checks ------------------------------------------------------------
    def check(self, case, quantity, got, ref, tol, scale=None):
        got, ref = np.asarray(got, dtype=complex), np.asarray(ref, dtype=complex)
        if scale is None:
            scale = max(float(np.max(np.abs(ref))), 1e-300)
        err = float(np.max(np.abs(got - ref))) / scale
        self.report.checks.append(Check(case, quantity, err, tol))

    def common(self, case, run, q, dim, ehrenfest=(1e-4, 1e-3), energy_tol=2e-5):
        """zkit/deck consistency, norm, <1/m> and the identities every run obeys."""
        ph, t = run.physics, run.time
        h = t[1] - t[0]
        dp, en = run.dipoles, run.energies
        s = -q
        self.check(case, "zkit carrier sign = -Charge", [ph.carrier_sign], [s], 0.0, 1.0)
        E = np.stack([np.broadcast_to(ph.field(a, t), t.shape) for a in range(dim)], 1)
        self.check(case, "field columns = zkit E(t)", dp[:, 1 : 1 + dim], E, 1e-12, 1.0)
        self.check(case, "Norm = 1", en[:, 6], np.ones_like(t), 1e-9)
        D = dp[:, 1 + dim : 1 + 2 * dim]
        A = dp[:, 1 + 2 * dim : 1 + 3 * dim]
        J = run.currents[:, 3 : 3 + dim]
        jsc, asc = max(np.max(np.abs(J)), 1e-3), max(np.max(np.abs(A)), 1e-3)
        self.check(case, "J = dD/dt", J[2:-2], time_derivative(D, h, 1), ehrenfest[0], jsc)
        self.check(case, "A = d2D/dt2", A[2:-2], time_derivative(D, h, 2), ehrenfest[1], asc)
        work = field_work(
            t, D / q, s, [lambda tt, a=a: ph.field_derivative(a, tt) for a in range(dim)]
        )
        tot = en[:, 4]
        self.check(
            case, "Tot - Tot(0) = int sE'.<r> dt", tot - tot[0], work, energy_tol,
            max(float(np.max(np.abs(tot - tot[0]))), 1e-3),
        )  # fmt: skip
        return t, s, E, D, A, J

    # -- cases ---------------------------------------------------------------
    def free_packet_3d(self, q: float):
        import sympy as sp

        m, w = 0.7, np.array([1.0, 1.2, 0.9])
        k0, x0 = np.array([0.5, -0.3, 0.2]), np.array([1.0, -0.5, 0.3])
        case = f"free packet 3D, m=0.7, q={q:+.0f}"
        text = f"""Verbose = 0
Dimension = 3
Domain = -14.0, 14.0
SplineDegree = 4
NQuadratures = 6
NSplines = 36
Charge = {q}
Mass = {m}
Potential = 0
PotentialDerivativeX = 0
PotentialDerivativeY = 0
PotentialDerivativeZ = 0
NBoundStates = 0
InitialState = wavepacket
WavepacketCenter = {x0[0]}, {x0[1]}, {x0[2]}
WavepacketMomentum = {k0[0]}, {k0[1]}, {k0[2]}
WavepacketWidth = {w[0]}, {w[1]}, {w[2]}
Polarization = xyz
LaserX = 0.05*sin(t)
LaserY = 0.03*cos(t)
LaserZ = 0.02
Propagator = pade22
TimeStep = 0.05
FinalTime = 3
OutputStrideWFS = 100000
"""
        run = self.run(f"free3d_q{q:+.0f}", text)
        # <r> is exact to ~3e-5 on this grid and time step (checked below); the
        # second difference of D and the energy integral inherit that error
        t, s, E, D, A, J = self.common(case, run, q, 3, ehrenfest=(1e-4, 3e-3), energy_tol=1e-4)
        ph, tau = run.physics, sp.Symbol("t")
        int_E, int2_E = [], []
        for a in range(3):
            e1 = sp.integrate(ph.field_expr(a), (tau, 0, tau))
            int_E.append(sp.lambdify(tau, e1, "numpy"))
            int2_E.append(sp.lambdify(tau, sp.integrate(e1, (tau, 0, tau)), "numpy"))
        int_E = [lambda tt, f=f: np.broadcast_to(f(tt), np.shape(tt)) for f in int_E]
        int2_E = [lambda tt, f=f: np.broadcast_to(f(tt), np.shape(tt)) for f in int2_E]
        r, v = free_packet(t, m, k0, x0, s, int_E, int2_E)
        en, c = run.energies, run.currents
        kin = np.sum(m * v * v / 2 + 1 / (8 * m * w[None, :] ** 2), 1)
        inter = np.sum(s * E * r, 1)
        self.check(case, "D = q<r> (exact)", D, q * r, 1e-4)
        self.check(case, "J = q<v> (exact)", J, q * v, 1e-4)
        self.check(case, "A = -q s E/m (exact)", A, -q * s * E / m, 1e-10)
        self.check(case, "Kin (exact)", en[:, 1], kin, 1e-4)
        self.check(case, "Pot = 0", en[:, 2], 0 * t, 1e-12, 1.0)
        self.check(case, "Int = s E.<r>", en[:, 3], inter, 1e-4)
        self.check(case, "Tot = Kin + Int", en[:, 4], kin + inter, 1e-4)
        self.check(case, "InvMass/Norm = 1/m", en[:, 5] / en[:, 6], 0 * t + 1 / m, 1e-12)
        self.check(case, "Lz, dGamma = 0", c[:, 1:3], 0 * c[:, 1:3], 1e-14, 1.0)
        self.check(case, "decomposition off (no states)", c[:, 6:15], 0 * c[:, 6:15], 1e-14, 1.0)
        ac = run.autocorrelation
        acx = free_packet_autocorrelation(t, m, w, k0, x0, s, int_E)
        self.check(case, "autocorrelation (exact, complex)", ac[:, 1] + 1j * ac[:, 2], acx, 1e-4)

    def driven_oscillator_2d(self, q: float):
        m, om = 1.5, 0.6
        s, k = -q, m * om * om
        case = f"driven oscillator 2D, m=1.5, q={q:+.0f}"
        text = f"""Verbose = 0
Dimension = 2
Domain = -12.0, 12.0
SplineDegree = 6
NQuadratures = 8
NSplines = 50
Charge = {q}
Mass = {m}
Potential = {s * 0.5 * k}*(x*x + y*y)
PotentialDerivativeX = {s * k}*x
PotentialDerivativeY = {s * k}*y
TargetEigenvalue = 0.0
NBoundStates = 1
InitialState = ground
Polarization = xy
LaserX = 0.08*sin(1.1*t)*sin(pi*t/20)^2
LaserY = 0.05*cos(0.7*t)*sin(pi*t/20)^2
Propagator = pade22
TimeStep = 0.02
FinalTime = 20
OutputStrideWFS = 100000
"""
        run = self.run(f"oscillator2d_q{q:+.0f}", text)
        t, s, E, D, A, J = self.common(case, run, q, 2)
        ph = run.physics
        self.check(
            case, "zkit V = s*Potential (confining)", [ph.V(1.0, 0.5)], [0.5 * k * 1.25], 1e-12
        )
        X, P, amp = coherent_state(t, m, om, s, [lambda tt, a=a: ph.field(a, tt) for a in range(2)])
        en, c = run.energies, run.currents
        kin = np.sum(om / 4 + P * P / (2 * m), 1)
        pot = np.sum(om / 4 + 0.5 * k * X * X, 1)
        inter = np.sum(s * E * X, 1)
        self.check(case, "E0 = w (eigensolver)", run.spectrum.values[:1], [om], 1e-9)
        self.check(case, "D = q<x> (exact)", D, q * X, 5e-5)
        self.check(case, "J = q<p>/m (exact)", J, q * P / m, 5e-5)
        self.check(case, "A = q(-w^2 x - sE/m) (exact)", A, q * (-(om**2) * X - s * E / m), 5e-5)
        self.check(case, "Kin (exact)", en[:, 1], kin, 5e-5)
        self.check(case, "Pot (exact)", en[:, 2], pot, 5e-5)
        self.check(case, "Int = s E.<r>", en[:, 3], inter, 5e-5)
        self.check(case, "Tot (exact)", en[:, 4], kin + pot + inter, 5e-5)
        self.check(case, "InvMass/Norm = 1/m", en[:, 5] / en[:, 6], 0 * t + 1 / m, 1e-12)
        self.check(case, "J_intra, J_inter = 0 (real state)", c[:, 5:9], 0 * c[:, 5:9], 1e-10, 1.0)
        self.check(case, "J_bc = J", c[:, 9:11], c[:, 3:5], 1e-10, max(np.max(np.abs(J)), 1e-3))
        self.check(case, "P_0 = |<0|psi>|^2 (exact)", run.populations[:, 1], np.abs(amp) ** 2, 5e-5)
        ac = run.autocorrelation[:, 1] + 1j * run.autocorrelation[:, 2]
        self.check(case, "autocorrelation (exact, complex)", ac, amp, 5e-5)
        gam = np.cumsum(c[:, 2])
        ref = np.exp(-1j * (np.unwrap(np.angle(amp)) - np.angle(amp[0])))
        self.check(case, "Berry phase: sum dGamma = -arg c0", np.exp(1j * gam), ref, 5e-5)

    def variable_mass(self, dim: int):
        ax = "xyz"[:dim]
        mass = "1 + 0.3*tanh(x/2) + 0.2*exp(-y*y/3)" + (" + 0.1*exp(-z*z/4)" if dim == 3 else "")
        pot = "0.15*(x*x + 1.5*y*y" + (" + 2*z*z" if dim == 3 else "") + ")"
        dpot = ["0.3*x", "0.45*y", "0.6*z"][:dim]
        head = (
            f"Verbose = 0\nDimension = {dim}\nDomain = -9.0, 9.0\nSplineDegree = 4\n"
            f"NQuadratures = 6\nNSplines = {40 if dim == 2 else 28}\nCharge = -1.0\n"
            f"Mass = {mass}\nPotential = {pot}\n"
            + "".join(f"PotentialDerivative{c.upper()} = {g}\n" for c, g in zip(ax, dpot))
            + f"TargetEigenvalue = 0.0\nNBoundStates = 1\nInitialState = ground\n"
            f"Polarization = {ax}\nPropagator = pade22\nTimeStep = 0.02\nOutputStrideWFS = 100000\n"
        )
        drive = "".join(
            f"Laser{c.upper()} = {a}*sin({wl}*t)\n"
            for c, a, wl in zip(ax, [0.06, 0.04, 0.03], [0.9, 1.3, 0.7])
        )
        acc = {}
        for method in ("operator", "discrete"):
            case = f"variable mass {dim}D stationary ({method})"
            run = self.run(
                f"varmass{dim}d_stationary_{method}",
                head
                + "".join(f"Laser{c.upper()} = 0\n" for c in ax)
                + f"FinalTime = 4\nAccelerationMethod = {method}\n",
            )
            t = run.time
            e0 = run.spectrum.values[0]
            dp, en, c = run.dipoles, run.energies, run.currents
            D, A = dp[:, 1 + dim : 1 + 2 * dim], dp[:, 1 + 2 * dim : 1 + 3 * dim]
            self.check(case, "D(t) = D(0)", D - D[0], 0 * D, 1e-9, 1.0)
            # operator form: zero to the spatial discretisation error (h^4)
            self.check(case, "A = 0", A, 0 * A, 2e-6 if method == "operator" else 1e-9, 1.0)
            self.check(case, "J = 0", c[:, 3 : 3 + dim], 0 * c[:, 3 : 3 + dim], 1e-9, 1.0)
            self.check(case, "Tot = E0", en[:, 4], 0 * t + e0, 1e-9)
            self.check(
                case, "Kin, Pot constant", en[:, 1:3] - en[0, 1:3], 0 * en[:, 1:3], 1e-9, 1.0
            )
            self.check(case, "Norm = 1, P_0 = 1", [en[:, 6], run.populations[:, 1]], 1.0, 1e-10)
            ac = run.autocorrelation[:, 1] + 1j * run.autocorrelation[:, 2]
            self.check(case, "autocorrelation = exp(-i E0 t)", ac, np.exp(-1j * e0 * t), 1e-7)
        for method in ("operator", "discrete"):
            case = f"variable mass {dim}D driven ({method})"
            run = self.run(
                f"varmass{dim}d_driven_{method}",
                head + drive + f"FinalTime = 8\nAccelerationMethod = {method}\n",
            )
            self.common(case, run, -1.0, dim)
            acc[method] = run.dipoles[:, 1 + 2 * dim : 1 + 3 * dim]
        self.check(
            f"variable mass {dim}D driven", "A operator = A discrete",
            acc["operator"], acc["discrete"], 5e-5,
        )  # fmt: skip


TDM_CASES = {
    # name: (dimension, deck body, reference, tolerance for energies and |d_ij|)
    "tdm oscillator 1D": (
        1,
        """Domain = -16, 16
SplineDegree = 6
NQuadratures = 10
NSplines = 120
Potential = 0.5*0.25*x*x
""",
        ("oscillator", (0.5,)),
        1e-6,
    ),
    "tdm box 1D": (
        1,
        """Domain = -5, 5
SplineDegree = 6
NQuadratures = 10
NSplines = 120
BoundaryType = dirichlet
Potential = 0
""",
        ("box", 10.0),
        1e-6,
    ),
    "tdm oscillator 2D": (
        2,
        """Domain = -14, 14, -14, 14
SplineDegree = 6
NQuadratures = 10
NSplines = 50
Potential = 0.5*(0.36*x*x + 0.6889*y*y)
""",
        ("oscillator", (0.6, 0.83)),
        1e-5,
    ),
    "tdm oscillator 3D": (
        3,
        """Domain = -10, 10, -10, 10, -10, 10
SplineDegree = 5
NQuadratures = 8
NSplines = 24
Potential = 0.5*(0.36*x*x + 0.6889*y*y + 1.3689*z*z)
""",
        ("oscillator", (0.6, 0.83, 1.17)),
        2e-3,
    ),
}


def _tdm_deck(dim: int, body: str, n_states: int) -> str:
    axes = "xyz"[:dim]
    return (
        f"Verbose = 0\nDimension = {dim}\nKnotSequence = uniform\nMass = 1.0\n{body}"
        f"TargetEigenvalue = 0.0\nNBoundStates = {n_states}\nNBoundStatesSave = 1\n"
        f"SaveDipoleAxes = {axes}\nEnablePropagation = 0\nPolarization = x\n"
    )


def transition_dipoles(v: "Validator", case: str, n_states: int = 6) -> None:
    """Spectrum and transition dipoles of a TDM_CASES entry against the exact results;
    dipole_spectrum.x / recombination.x rows against dipoles.x on every axis."""
    import csv

    from zkit.analytic import box_tdm, compare_tdm, oscillator_tdm
    from zkit.io.tdm import read_tdm

    dim, body, (kind, par), tol = TDM_CASES[case]
    ref = oscillator_tdm(par, n_states) if kind == "oscillator" else box_tdm(par, n_states)
    name = case.replace(" ", "_")
    run = v._run(name, _tdm_deck(dim, body, n_states), v.binary)
    bindir = Path(v.binary).resolve().parent
    tool = lambda t: str(bindir / t)  # noqa: E731
    r = subprocess.run(
        [tool("dipoles.x"), "in.inp"], cwd=run.run_dir, capture_output=True, text=True
    )
    if r.returncode:
        raise RuntimeError(f"{case}: dipoles.x failed\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    tdm = read_tdm(run.path("eigen"))
    energies = run.spectrum.values
    errs = compare_tdm(tdm, energies, ref)
    v.report.checks.append(Check(case, "energies = exact", errs["energies"], tol))
    for a in "xyz"[:dim]:
        v.report.checks.append(Check(case, f"|<i|{a}|j>| = exact (dipoles.x)", errs[a], tol))
        d = tdm[f"tdm_{a}"]
        i0 = int(np.argmin(energies))  # ground state as bra
        for t, out in (
            ("dipole_spectrum.x", f"in_dij_D{a}_i{i0}.csv"),
            ("recombination.x", f"in_drec_D{a}.csv"),
        ):
            args = [tool(t), "in.inp", "-i", str(i0), "-Ethr", "-1e9", "-axis", a]
            r = subprocess.run(args, cwd=run.run_dir, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"{case}: {t} failed\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            with open(run.run_dir / "static" / out, newline="") as fh:
                rows = list(csv.DictReader(fh))
            if t == "dipole_spectrum.x":
                got = np.array([abs(complex(float(w["Re_d"]), float(w["Im_d"]))) for w in rows])
                want = np.abs(d[i0, [int(w["j"]) for w in rows]])
            else:  # rows sorted by energy; d_j column is the regular dipole
                got = np.array([abs(complex(float(w["Re_dj"]), float(w["Im_dj"]))) for w in rows])
                want = np.abs(d[i0, np.argsort(energies)])
            v.check(case, f"{t} row {a} = dipoles.x", got, want, 1e-10, max(np.max(want), 1e-300))


KH_CASES = {
    # name: (deck, frequencies of the oscillator, shift, tolerance of the absolute levels)
    "kh oscillator 1D, m = 2, linear": (
        """Dimension = 1
Domain = -20, 20
SplineDegree = 6
NQuadratures = 10
NSplines = 160
Mass = 2.0
Potential = 0.5*2.0*0.25*x*x
Polarization = x
LaserX = 0.3*sin(1.5*t)
FinalTime = 20
""",
        (0.5,),
        ("linear", 0.5, 0.3, 1.5, 2.0),
        1e-9,
    ),
    "kh oscillator 2D, circular": (
        """Dimension = 2
Domain = -14, 14, -14, 14
SplineDegree = 6
NQuadratures = 10
NSplines = 50
Potential = 0.5*0.49*(x*x + y*y)
Polarization = xy
LaserX = 0.4*sin(1.3*t)
LaserY = 0.4*cos(1.3*t)
FinalTime = 30
""",
        (0.7, 0.7),
        ("circular", 0.7, 0.4, 1.3, 1.0),
        1e-6,
    ),
    "kh oscillator 3D, linear z": (
        """Dimension = 3
Domain = -10, 10, -10, 10, -10, 10
SplineDegree = 5
NQuadratures = 8
NSplines = 22
Potential = 0.5*(0.36*x*x + 0.6889*y*y + 1.3689*z*z)
Polarization = z
LaserZ = 0.5*sin(2.0*t)
FinalTime = 20
""",
        (0.6, 0.83, 1.17),
        ("linear", 1.17, 0.5, 2.0, 1.0),
        3e-3,
    ),
}


def kramers_henneberger(v: "Validator", case: str, n_states: int = 5) -> None:
    """Mode = kh against the exact oscillator shift: KH minus lab-frame levels on
    the same grid (the discretisation cancels) and the absolute levels."""
    from zkit.analytic import kh_oscillator_shift, oscillator_states

    deck, omega, (pol, w0, e0, w, m), tol = KH_CASES[case]
    shift = kh_oscillator_shift(w0, e0, w, mass=m, circular=pol == "circular")
    head = f"Verbose = 0\nNBoundStates = {n_states}\nTargetEigenvalue = 0\n"
    name = case.replace(" ", "_").replace(",", "").replace("=", "")
    lab = np.sort(
        v._run(
            name + "_td", head + deck + "Mode = td\nEnablePropagation = 0\n", v.binary
        ).spectrum.values
    )
    kh = np.sort(v._run(name + "_kh", head + deck + "Mode = kh\n", v.binary).spectrum.values)
    n = min(len(lab), len(kh))
    v.report.checks.append(
        Check(
            case,
            "KH - lab levels = exact shift",
            float(np.max(np.abs(kh[:n] - lab[:n] - shift)) / shift),
            1e-7,
        )
    )
    exact = oscillator_states(omega, n)[1] + shift
    v.report.checks.append(
        Check(case, "KH levels = exact", float(np.max(np.abs(kh[:n] - exact) / exact)), tol)
    )


def validate(binary, reference=None, nproc=4, workdir="zkit-validation", quick=False) -> Report:
    """Run the validation suite; ``quick`` runs the 2D cases only (about a minute)."""
    v = Validator(binary, reference, nproc, workdir)
    for case in TDM_CASES:
        if quick and "3D" in case:
            continue
        transition_dipoles(v, case)
    for case in KH_CASES:
        if quick and "3D" in case:
            continue
        kramers_henneberger(v, case)
    for q in (-1.0, 1.0):
        v.driven_oscillator_2d(q)
        if quick:
            break
    v.variable_mass(2)
    if not quick:
        for q in (-1.0, 1.0):
            v.free_packet_3d(q)
        v.variable_mass(3)
    return v.report


__all__ = [
    "Check",
    "Report",
    "Validator",
    "coherent_state",
    "default_binary",
    "field_work",
    "free_packet",
    "free_packet_autocorrelation",
    "kramers_henneberger",
    "time_derivative",
    "transition_dipoles",
    "validate",
]

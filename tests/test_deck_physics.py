"""Deck reading and deck-derived physics follow the TDSE-Z rules."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from zkit.deck import expression_constants, read_deck

sympy = pytest.importorskip("sympy")

from zkit.physics import parse_expression  # noqa: E402
from zkit.simulation import Run  # noqa: E402

H2P_LASER = (
    "((t < 2*pi/Omegax) ? (t*Omegax/(2*pi)) : ((t < 18*pi/Omegax) ? 1.0 : "
    "((t <= 20*pi/Omegax) ? (10.0 - t*Omegax/(2*pi)) : 0.0))) * Ampx * cos(Omegax*t + CEPx)"
)


def _trapezoid(t, w, a):
    env = np.where(
        t < 2 * np.pi / w,
        t * w / (2 * np.pi),
        np.where(
            t < 18 * np.pi / w, 1.0, np.where(t <= 20 * np.pi / w, 10.0 - t * w / (2 * np.pi), 0.0)
        ),
    )
    return env * a * np.cos(w * t)


def _write(tmp_path: Path, text: str, name: str = "deck.inp") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_nested_ternary_matches_direct_evaluation():
    consts = {"Omegax": 0.05655, "Ampx": 0.02, "CEPx": 0.0, "pi": np.pi}
    f = sympy.lambdify(sympy.Symbol("t"), parse_expression(H2P_LASER, ("t",), consts), "numpy")
    t = np.linspace(0.0, 1300.0, 4001)
    assert np.allclose(f(t), _trapezoid(t, 0.05655, 0.02), atol=1e-15)


def test_right_associative_ternary_without_parentheses():
    e = parse_expression("x < 0 ? -1 : x < 1 ? 0 : 1", ("x",))
    f = sympy.lambdify(sympy.Symbol("x"), e, "numpy")
    assert [float(f(v)) for v in (-2.0, 0.5, 3.0)] == [-1.0, 0.0, 1.0]


def test_deck_rules(tmp_path):
    path = _write(
        tmp_path,
        """
Dimension = 2   # inline comment
NSplines = 40
TimeStepSize = 0.05
TotalTime = 100   # alias of FinalTime
Domain = [-20,20],[-5,5]
LaserAmplitude = 0.01, 0.02
Variables = A0=1.5
Variables = B0=2.0, C0=3
DriverX = Ampx*sin(t)
""",
    )
    params, variables = read_deck(path)
    assert params["dimension"] == "2"
    assert (
        params["nelements"] == "40"
        and params["timestep"] == "0.05"
        and params["finaltime"] == "100"
    )
    assert (params["lminy"], params["lmaxy"], params["lminz"]) == ("-5", "5", "-20")
    assert (params["ampx"], params["ampy"], params["ampz"]) == ("0.01", "0.02", "0.01")
    assert params["laserx"] == "Ampx*sin(t)"
    assert variables == {"A0": 1.5, "B0": 2.0, "C0": 3.0}
    c = expression_constants(params, variables)
    assert c["Ampy"] == 0.02 and c["NSplines"] == 40 and c["Charge"] == 1.0 and c["C0"] == 3.0


def test_physics_field_follows_tdsez(tmp_path):
    _write(
        tmp_path,
        f"""
Dimension = 2
TimeStep = 0.5
FinalTime = 1500
Omegax = 0.05655
Ampx = 0.02
Polarization = xy
LaserX = {H2P_LASER}
Laser = 0.5*Ampx*sin(Omegax*t)
Potential = V0 + x*x + y*y
Variables = V0=1.0
""",
        "h2p.inp",
    )
    phys = Run(tmp_path, "h2p.inp").physics
    t = np.linspace(0, 1400, 2801)
    assert np.allclose(phys.Ex(t), _trapezoid(t, 0.05655, 0.02), atol=1e-15)
    assert np.allclose(phys.Ey(t), 0.5 * 0.02 * np.sin(0.05655 * t))  # Laser fills y
    assert float(phys.Ez(1.0)) == 0.0  # z is not polarised
    assert phys.laser_on_until == pytest.approx(1500.0)  # the y field never switches off
    assert float(phys.V(1.0, 2.0)) == pytest.approx(6.0)


def test_laser_end_from_field(tmp_path):
    _write(
        tmp_path,
        f"""
Dimension = 1
TimeStep = 0.5
FinalTime = 1500
Omegax = 0.05655
Ampx = 0.02
Polarization = x
LaserX = {H2P_LASER}
""",
        "pulse.inp",
    )
    phys = Run(tmp_path, "pulse.inp").physics
    assert phys.laser_on_until == pytest.approx(20 * np.pi / 0.05655, abs=0.5)


def test_no_field_is_invented_from_amplitudes(tmp_path):
    _write(
        tmp_path,
        "Dimension = 1\nAmpx = 0.05\nOmegax = 0.057\nPolarization = x\nFinalTime = 10\n",
        "a.inp",
    )
    phys = Run(tmp_path, "a.inp").physics
    assert not phys.has_laser and float(phys.Ex(1.0)) == 0.0 and phys.laser_on_until is None


def test_time_axis_when_dipoles_disabled(tmp_path):
    from zkit.io.evolution import read_evolution

    n = 5
    t = np.arange(n) * 0.1
    with h5py.File(tmp_path / "TimeEvolutionData_x.inp.h5", "w") as f:
        f["dipoles"] = np.zeros((n, 4))
        f["populations"] = np.column_stack([t, np.ones(n)])
        f["energies"] = np.column_stack([t, np.zeros((n, 6))])
        f["currents"] = np.zeros((n, 11))
        f["autocorrelation"] = np.zeros((n, 3))
    ev = read_evolution(tmp_path / "TimeEvolutionData_x.inp.h5")
    assert np.allclose(ev.time, t)

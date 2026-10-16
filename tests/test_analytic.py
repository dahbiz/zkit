"""zkit.analytic against direct quadrature, and read_tdm on dipoles.x output."""

import h5py
import numpy as np
import pytest
from numpy.polynomial.hermite import hermval
from scipy.special import factorial

from zkit.analytic import (
    box_tdm,
    compare_tdm,
    is_nondegenerate,
    oscillator_states,
    oscillator_tdm,
)
from zkit.io.tdm import read_tdm


def _ho_1d(n, w, x):
    c = np.zeros(n + 1)
    c[n] = 1
    norm = (w / np.pi) ** 0.25 / np.sqrt(2.0**n * factorial(n))
    return norm * hermval(np.sqrt(w) * x, c) * np.exp(-w * x * x / 2)


def test_oscillator_dipoles_equal_quadrature() -> None:
    w = 0.7
    x = np.linspace(-14, 14, 40001)
    ref = oscillator_tdm((w,), 6)
    for i in range(6):
        for j in range(6):
            d = np.trapezoid(_ho_1d(i, w, x) * x * _ho_1d(j, w, x), x)
            assert abs(d) == pytest.approx(abs(ref["x"][i, j]), abs=1e-10)


def test_box_dipoles_equal_quadrature() -> None:
    L, a = 7.0, -2.0
    x = np.linspace(a, a + L, 200001)
    phi = lambda n: np.sqrt(2 / L) * np.sin(n * np.pi * (x - a) / L)  # noqa: E731
    ref = box_tdm(L, 5, start=a)
    for n in range(1, 6):
        for m in range(1, 6):
            d = np.trapezoid(phi(n) * x * phi(m), x)
            assert d == pytest.approx(ref["x"][n - 1, m - 1], abs=1e-8)


def test_oscillator_states_and_degeneracy() -> None:
    qn, e = oscillator_states((0.6, 0.83, 1.17), 4)
    assert qn.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert e[0] == pytest.approx(0.5 * (0.6 + 0.83 + 1.17))
    assert not is_nondegenerate(oscillator_tdm((0.6, 0.9), 6))  # (3,0) and (0,2) at 2.55
    assert is_nondegenerate(oscillator_tdm((0.6, 0.83), 6))
    with pytest.raises(ValueError):
        compare_tdm({"x": np.eye(6)}, np.arange(6.0), oscillator_tdm((0.6, 0.9), 6))


@pytest.mark.parametrize("layout", ["current", "former"])
def test_read_tdm_finds_dipoles_output(tmp_path, layout) -> None:
    ref = oscillator_tdm((0.6, 0.83), 5)
    static = tmp_path / "static"
    static.mkdir()
    if layout == "current":
        eig, npy = static / "run_eigen.h5", {a: static / f"run_tdm_D{a}.npy" for a in "xy"}
    else:
        eig, npy = (
            static / "EigenData_run.inp.h5",
            {a: static / f"TDM_D{a}_run.inp.npy" for a in "xy"},
        )
    with h5py.File(eig, "w") as f:
        f["spectrum"] = np.stack([ref["energies"], 0 * ref["energies"]], 1)
    perm = np.array([2, 0, 4, 1, 3])  # stored in another order than the energies
    for a in "xy":
        np.save(npy[a], (-ref[a][np.ix_(perm, perm)]).astype(complex))  # any phase
    tdm = read_tdm(eig)
    assert sorted(tdm) == ["tdm_x", "tdm_y"]
    err = compare_tdm(tdm, ref["energies"][perm], ref)
    assert err["x"] < 1e-14 and err["y"] < 1e-14 and err["energies"] < 1e-14

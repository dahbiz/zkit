"""Analytic references of zkit.validation against independent closed forms.

These tests need no TDSE-Z binary. test_quick_suite runs the 2D validation
cases end to end when TDSEZ_BINARY (or tdsez.x on PATH) is available.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sympy")

import zkit  # noqa: E402
from zkit import validation as V  # noqa: E402


def test_time_derivative_is_fourth_order():
    t = np.linspace(0, 2, 201)
    h = t[1] - t[0]
    y = np.sin(1.3 * t)[:, None]
    assert np.max(np.abs(V.time_derivative(y, h, 1)[:, 0] - 1.3 * np.cos(1.3 * t[2:-2]))) < 1e-8
    assert np.max(np.abs(V.time_derivative(y, h, 2)[:, 0] + 1.69 * y[2:-2, 0])) < 1e-6


def test_free_packet_without_field_moves_uniformly():
    t = np.linspace(0, 3, 31)
    zero = [lambda tt: 0 * tt] * 3
    r, v = V.free_packet(t, 0.7, [0.5, -0.3, 0.2], [1.0, -0.5, 0.3], 1.0, zero, zero)
    assert np.allclose(r[:, 0], 1.0 + 0.5 * t / 0.7)
    assert np.allclose(v, np.array([0.5, -0.3, 0.2]) / 0.7)


def test_free_packet_in_constant_field():
    # H = p^2/2m + s E x: <x> = x0 + k0 t/m - s E t^2/(2m)
    t = np.linspace(0, 2, 21)
    E = 0.1
    r, v = V.free_packet(
        t, 2.0, [0.3], [0.0], -1.0, [lambda tt: E * tt], [lambda tt: E * tt**2 / 2]
    )
    assert np.allclose(r[:, 0], 0.3 * t / 2.0 + E * t**2 / 4.0)
    assert np.allclose(v[:, 0], (0.3 + E * t) / 2.0)


def test_free_packet_autocorrelation_matches_closed_form():
    # without a field: prod_a (1 + i t/4mw^2)^(-1/2) exp(-i k0^2 t/2m / (1 + i t/4mw^2))
    m, w, k0, x0 = (
        0.7,
        np.array([1.0, 1.2, 0.9]),
        np.array([0.5, -0.3, 0.2]),
        np.array([1, -0.5, 0.3]),
    )
    t = np.linspace(0, 3, 13)
    zero = [lambda tt: 0 * tt] * 3
    ac = V.free_packet_autocorrelation(t, m, w, k0, x0, 1.0, zero, n_tau=3001)
    b = t[:, None] / (4 * m * w[None, :] ** 2)
    exact = np.prod(
        (1 + 1j * b) ** -0.5 * np.exp(-1j * k0**2 * t[:, None] / (2 * m) / (1 + 1j * b)), 1
    )
    assert abs(ac[0] - 1) < 1e-10
    assert np.max(np.abs(ac - exact)) < 1e-8


def test_coherent_state_without_field_is_the_ground_state():
    t = np.linspace(0, 10, 101)
    X, P, amp = V.coherent_state(t, 1.5, 0.6, 1.0, [lambda tt: 0 * tt] * 2)
    assert np.max(np.abs(X)) < 1e-12 and np.max(np.abs(P)) < 1e-12
    assert np.max(np.abs(amp - np.exp(-1j * 0.6 * t))) < 1e-9  # E0 = w in 2D


def test_coherent_state_follows_the_classical_oscillator():
    m, om, s = 1.5, 0.6, -1.0
    t = np.linspace(0, 20, 2001)

    def E(tt):
        return 0.08 * np.sin(1.1 * tt) * np.sin(np.pi * tt / 20) ** 2

    X, P, amp = V.coherent_state(t, m, om, s, [E])
    h = t[1] - t[0]
    acc = V.time_derivative(X, h, 2)[:, 0]
    assert np.max(np.abs(acc - (-(om**2) * X[2:-2, 0] - s * E(t[2:-2]) / m))) < 1e-7
    assert np.allclose(P[:, 0], m * np.gradient(X[:, 0], t), atol=1e-4)
    # |<0|alpha>|^2 = exp(-|alpha|^2), |alpha|^2 = (m w x^2 + p^2/(m w))/2
    alpha2 = (m * om * X[:, 0] ** 2 + P[:, 0] ** 2 / (m * om)) / 2
    assert np.allclose(np.abs(amp) ** 2, np.exp(-alpha2), atol=1e-10)


def test_field_work():
    t = np.linspace(0, 4, 401)
    r = np.stack([np.cos(t), 0 * t], 1)
    work = V.field_work(t, r, -1.0, [lambda tt: np.cos(tt), lambda tt: 0 * tt])
    exact = -(t / 2 + np.sin(2 * t) / 4)  # int_0^t s cos^2 with s = -1
    assert np.max(np.abs(work - exact)) < 1e-8


def test_physics_field_derivative(tmp_path):
    (tmp_path / "in.inp").write_text(
        "Dimension = 2\nPolarization = x\nLaserX = 0.05*sin(2*t)*(t < 3 ? 1 : 0)\n"
    )
    ph = zkit.Run(tmp_path, "in.inp").physics
    t = np.linspace(0.1, 2.9, 15)
    assert np.allclose(ph.field(0, t), 0.05 * np.sin(2 * t))
    assert np.allclose(ph.field_derivative("x", t), 0.1 * np.cos(2 * t))
    assert np.allclose(ph.field_derivative(1, t), 0.0)  # y is not coupled


@pytest.mark.skipif(V.default_binary() is None, reason="no TDSE-Z binary (set TDSEZ_BINARY)")
def test_quick_suite(tmp_path):
    report = V.validate(V.default_binary(), nproc=2, workdir=tmp_path, quick=True)
    print(report.text())
    assert report.ok

"""Exact eigenstates and transition dipoles of reference systems.

Atomic units, hbar = 1. These are the references for TDSE-Z's eigensolver and
for the transition dipoles d_ij = <i|x_a|j> computed by ``dipoles.x``
(:func:`zkit.io.tdm.read_tdm`).

Harmonic oscillator, V = sum_a m w_a^2 x_a^2 / 2 (1D, 2D, 3D), product states
|n_x n_y n_z> with E = sum_a w_a (n_a + 1/2) and

    <n|x_a|n+1> = sqrt((n + 1) / (2 m w_a))   (all other quantum numbers equal),

zero otherwise. With distinct frequencies the low-lying levels are
non-degenerate, so states can be matched by energy and |d_ij| compared
independently of the arbitrary eigenvector phases.

Static Kramers-Henneberger frame (TDSE-Z Mode = kh): for an oscillator the
cycle-averaged potential is V_KH = V + m w0^2 <|alpha|^2> / 2, so every level
shifts by m w0^2 alpha0^2 / 4 for a linear field (alpha = alpha0 sin wt) and
m w0^2 alpha0^2 / 2 for a circular one (|alpha| = alpha0), with the quiver
amplitude alpha0 = |q| E0 / (m w^2) (:func:`kh_oscillator_shift`).

Particle in a box [a, a + L] (hard walls): E_n = pi^2 n^2 / (2 m L^2) and,
for n != m,

    <n|x|m> = -8 L n m / (pi^2 (n^2 - m^2)^2)   if n + m is odd, else 0.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def oscillator_states(omega: Sequence[float], n_states: int) -> tuple:
    """The ``n_states`` lowest product states of an (anisotropic) oscillator.

    Returns (quantum numbers, (n_states, dim) int array; energies, (n_states,)).
    Ties are broken by the quantum numbers, so check :func:`is_nondegenerate`
    before matching numerical states by energy.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    dim = len(omega)
    nmax = int(np.ceil(n_states)) + 1
    qn = np.array(list(itertools.product(range(nmax), repeat=dim)), dtype=int)
    energy = (qn + 0.5) @ omega
    order = np.lexsort((*qn.T[::-1], energy))[:n_states]
    return qn[order], energy[order]


def oscillator_tdm(omega: Sequence[float], n_states: int, mass: float = 1.0) -> dict:
    """Exact dipole matrices <i|x_a|j> (real, phase convention of the ladder
    operators) between the ``n_states`` lowest oscillator states.

    Returns {"energies": (N,), "states": (N, dim), "x": (N, N), "y": ..., "z": ...}
    with one matrix per axis of ``omega``.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    qn, energy = oscillator_states(omega, n_states)
    nxt = oscillator_states(omega, n_states + 1)[1][-1]
    out = {"energies": energy, "states": qn, "next_energy": float(nxt)}
    for a, w in enumerate(omega):
        d = np.zeros((n_states, n_states))
        for i, j in itertools.product(range(n_states), repeat=2):
            diff = qn[j] - qn[i]
            others = np.delete(diff, a)
            if abs(diff[a]) == 1 and not others.any():
                n = max(qn[i][a], qn[j][a])
                d[i, j] = np.sqrt(n / (2.0 * mass * w))
        out["xyz"[a]] = d
    return out


def box_energies(length: float, n_states: int, mass: float = 1.0) -> np.ndarray:
    """E_n = pi^2 n^2 / (2 m L^2), n = 1 .. n_states."""
    n = np.arange(1, n_states + 1)
    return np.pi**2 * n**2 / (2.0 * mass * length**2)


def box_tdm(length: float, n_states: int, start: float = None) -> dict:
    """Exact <n|x|m> of a hard-wall box of width ``length`` (box [start, start + L],
    centred at 0 by default), for n, m = 1 .. n_states.

    Returns {"energies": (N,), "x": (N, N)}.
    """
    if start is None:
        start = -0.5 * length
    n = np.arange(1, n_states + 1)
    N, M = np.meshgrid(n, n, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -8.0 * length * N * M / (np.pi**2 * (N**2 - M**2) ** 2)
    d = np.where((N + M) % 2 == 1, d, 0.0)
    np.fill_diagonal(d, start + 0.5 * length)
    return {
        "energies": box_energies(length, n_states),
        "x": d,
        "next_energy": float(box_energies(length, n_states + 1)[-1]),
    }


def is_nondegenerate(reference, gap: float = 1e-6) -> bool:
    """Whether the reference levels, and the first level above them, are all
    separated by more than ``gap`` (a reference dict or an array of energies)."""
    if isinstance(reference, dict):
        e = np.append(reference["energies"], reference.get("next_energy", np.inf))
    else:
        e = np.asarray(reference, dtype=float)
    return bool(np.all(np.diff(np.sort(e)) > gap))


def compare_tdm(numeric: dict, numeric_energies: np.ndarray, reference: dict) -> dict:
    """Compare numerical transition dipoles with a reference by matching states
    in energy order (both sets non-degenerate).

    ``numeric`` maps an axis (``"x"``, ``"tdm_x"``, ...) to an (N, N) complex
    matrix; ``reference`` is the output of :func:`oscillator_tdm` or
    :func:`box_tdm`. Off-diagonal magnitudes |d_ij| are compared (eigenvector
    phases are arbitrary). Returns {axis: max |(|d| - |d_ref|)| / max |d_ref|},
    plus ``"energies"``: the max relative energy error.
    """
    if not is_nondegenerate(reference):
        raise ValueError("the reference levels are degenerate: states cannot be matched by energy")
    e_num = np.asarray(numeric_energies, dtype=float)
    n = min(len(e_num), len(reference["energies"]))
    order = np.argsort(e_num)[:n]
    out = {
        "energies": float(
            np.max(
                np.abs(e_num[order] - reference["energies"][:n]) / np.abs(reference["energies"][:n])
            )
        )
    }
    off = ~np.eye(n, dtype=bool)
    for key, mat in numeric.items():
        axis = key[-1]
        if axis not in reference:
            continue
        got = np.abs(np.asarray(mat)[np.ix_(order, order)])[off]
        ref = np.abs(reference[axis][:n, :n])[off]
        out[axis] = float(np.max(np.abs(got - ref)) / max(np.max(ref), 1e-300))
    return out


def kh_oscillator_shift(
    omega0: float,
    e0: float,
    omega: float,
    mass: float = 1.0,
    charge: float = -1.0,
    circular: bool = False,
) -> float:
    """Level shift of an oscillator (frequency ``omega0`` along the field) in the
    static Kramers-Henneberger frame, for a field of amplitude ``e0`` and
    frequency ``omega``: m w0^2 alpha0^2 / 4 (linear) or / 2 (circular),
    alpha0 = |q| e0 / (m omega^2)."""
    alpha0 = abs(charge) * e0 / (mass * omega**2)
    return (0.5 if circular else 0.25) * mass * omega0**2 * alpha0**2


__all__ = [
    "box_energies",
    "kh_oscillator_shift",
    "box_tdm",
    "compare_tdm",
    "is_nondegenerate",
    "oscillator_states",
    "oscillator_tdm",
]

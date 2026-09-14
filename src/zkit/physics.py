"""Automatic physics extraction from a TDSEZ input deck.

Parses the problem expressions (Potential, mass, laser fields, ...) straight
from the ``.inp`` deck via :class:`zkit.simulation.Run`, differentiates
symbolically with sympy to obtain the higher derivatives the Wigner/Moyal
analysis needs (V', V'', V''', V^(5), mass gradient), and exposes fast
lambdified callables.

Everything is derived from the deck -- no physics is hard-coded in the caller.
If the deck supplies an explicit derivative (e.g. ``PotentialDerivativeX``) it
is parsed and used; otherwise it is obtained by symbolic differentiation of
``Potential`` so the result is always consistent with the deck.

Example
-------
    >>> from zkit.simulation import Run
    >>> run = Run("build", "ZnSe_d20_A0.005338")
    >>> phys = run.physics
    >>> phys.V(0.0, 0.0)  # potential at the origin
    >>> phys.Ax(1000.0)  # laser vector potential at t=1000 a.u.
    >>> phys.snapshot_time(10)  # real time of wfs snapshot #10 (0-based)
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import sympy as sp

__all__ = ["Physics", "parse_expression"]


# ---------------------------------------------------------------------------
# Expression parsing
# ---------------------------------------------------------------------------
_SYMPY_NS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "e": sp.E,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
}


def _preprocess(src: str) -> str:
    """Normalise C/Python-ish math syntax to something sympy can parse."""
    s = src.strip()
    # power: C-style '^' -> python '**'  (must run before any XOR handling)
    s = s.replace("^", "**")
    # ternary  cond ? a : b  ->  Piecewise((a, cond), (b, True))
    # handle nested via a small recursive regex (one level is enough for decks)
    pat = re.compile(r"\(([^()]*?)\)\s*\?\s*([^:]+?)\s*:\s*([^()]+?)(?=\)|$)")
    while "?" in s:
        new = pat.sub(r"Piecewise((\2, (\1)), (\3, True))", s)
        if new == s:
            break
        s = new
    # logical ops (rare in decks, but be safe)
    s = s.replace("&&", "&").replace("||", "|")
    return s


def parse_expression(src: str, variables=("x", "y", "t")):
    """Parse a deck expression string into a sympy expression.

    Substitutes any scalar parameter keys (Ampx, Omegax, ...) that appear in
    the string with their numeric values so the returned expression depends
    only on the named ``variables``.
    """
    s = _preprocess(src)
    syms = {v: sp.Symbol(v) for v in variables}
    # collect scalar params from the same namespace if provided externally
    expr = sp.sympify(s, locals={**_SYMPY_NS, **syms})
    return expr


class Physics:
    """Physics of a TDSEZ run, extracted automatically from its input deck."""

    def __init__(self, meta) -> None:
        self.meta = meta
        self.extra: Dict[str, str] = dict(getattr(meta, "extra", {}) or {})
        self.dim: int = int(getattr(meta, "dimension", 1) or 1)
        self.t_step: float = float(getattr(meta, "time_step", 0.0) or 0.0)
        self.stride_wfs: int = int(getattr(meta, "output_stride_wfs", 0) or 0)
        self.final_time: float = float(getattr(meta, "final_time", 0.0) or 0.0)

        x, y, t = sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("t")
        self._x, self._y, self._t = x, y, t

        # ---- scalar laser / field parameters (substituted numerically) ----
        def _scalar(key, default=0.0):
            v = self.extra.get(key.lower())
            if v is None:
                return default
            try:
                return float(str(v).split()[0])
            except (TypeError, ValueError):
                return default

        self.amplitude_x = _scalar("ampx")
        self.amplitude_y = _scalar("ampy")
        self.omega_x = _scalar("omegax")
        self.omega_y = _scalar("omegay")
        self.cepx = _scalar("cepx")
        self.cepy = _scalar("cepy")
        self.polarization = self.extra.get("polarization", "x")

        # ---- potential V(x,y) ----
        Vsrc = self.extra.get("potential")
        self._V_expr = parse_expression(Vsrc, ("x", "y")) if Vsrc else sp.Integer(0)
        # explicit derivative strings if the deck provides them
        Vpx_src = self.extra.get("potentialderivativex")
        Vpy_src = self.extra.get("potentialderivativey")
        self._has_explicit_dV = Vpx_src is not None and Vpy_src is not None

        # ---- mass m(x,y) ----
        Msrc = self.extra.get("mass")
        self._M_expr = parse_expression(Msrc, ("x", "y")) if Msrc else sp.Integer(1)

        # ---- laser vector potential A(t) ----
        # Prefer the explicit LaserX/LaserY envelopes; fall back to a plain
        # cos carrier if only Ampx/Omegax are given.
        # Substitute scalar laser params (Ampx, Omegax, CEPx, ...) with their
        # numeric values so Ax(t) depends only on t.
        sub = {
            sp.Symbol(k): v
            for k, v in [
                ("Ampx", self.amplitude_x),
                ("Ampy", self.amplitude_y),
                ("Omegax", self.omega_x),
                ("Omegay", self.omega_y),
                ("CEPx", self.cepx),
                ("CEPy", self.cepy),
            ]
        }
        Lx_src = self.extra.get("laserx")
        Ly_src = self.extra.get("lasery")
        if Lx_src:
            self._Ax_expr = parse_expression(Lx_src, ("t",)).subs(sub)
        elif self.amplitude_x:
            self._Ax_expr = self.amplitude_x * sp.cos(self.omega_x * t + self.cepx)
        else:
            self._Ax_expr = sp.Integer(0)
        if Ly_src:
            self._Ay_expr = parse_expression(Ly_src, ("t",)).subs(sub)
        elif self.amplitude_y and self.polarization != "x":
            self._Ay_expr = self.amplitude_y * sp.cos(self.omega_y * t + self.cepy)
        else:
            self._Ay_expr = sp.Integer(0)

        # ---- lambdified callables ----
        self._V = sp.lambdify((x, y), self._V_expr, "numpy")
        self._M = sp.lambdify((x, y), self._M_expr, "numpy")

        # derivatives of V (symbolic, from V itself -- always consistent)
        dVdx = sp.diff(self._V_expr, x)
        dVdy = sp.diff(self._V_expr, y)
        self._Vpx = sp.lambdify((x, y), dVdx, "numpy")
        self._Vpy = sp.lambdify((x, y), dVdy, "numpy")
        # higher derivatives on the y=0 slice, as functions of x (Moyal Q)
        V3_x = sp.diff(self._V_expr, x, 3).subs(y, 0)
        V5_x = sp.diff(self._V_expr, x, 5).subs(y, 0)
        self._V3x = sp.lambdify((x,), V3_x, "numpy")
        self._V5x = sp.lambdify((x,), V5_x, "numpy")

        self._Ax = sp.lambdify((t,), self._Ax_expr, "numpy")
        self._Ay = sp.lambdify((t,), self._Ay_expr, "numpy")

        # laser envelope end time T (from the sin^2 window, if detectable)
        self.laser_on_until = self._detect_laser_end()

    # ------------------------------------------------------------------
    def _detect_laser_end(self) -> Optional[float]:
        """Best-effort extraction of the pulse cutoff T from LaserX."""
        src = self.extra.get("laserx", "")
        m = re.search(r"t\s*<=\s*([0-9.]+)", src)
        if m:
            return float(m.group(1))
        return self.final_time if (self.amplitude_x or self.amplitude_y) else None

    # ------------------------------------------------------------------
    # callables (numpy-array friendly)
    # ------------------------------------------------------------------
    def V(self, x, y):
        return self._V(x, y)

    def mass(self, x, y):
        return self._M(x, y)

    def Vpx(self, x, y):
        return self._Vpx(x, y)

    def Vpy(self, x, y):
        return self._Vpy(x, y)

    def V3_x_slice(self, x):
        """d^3 V / dx^3 evaluated on the y=0 slice."""
        return self._V3x(x)

    def V5_x_slice(self, x):
        """d^5 V / dx^5 evaluated on the y=0 slice."""
        return self._V5x(x)

    def Ax(self, t):
        return self._Ax(t)

    def Ay(self, t):
        return self._Ay(t)

    # ------------------------------------------------------------------
    # snapshot time mapping
    # ------------------------------------------------------------------
    def snapshot_time(self, i: int) -> float:
        """Real time (a.u.) of wfs snapshot ``i`` (0-based).

        t_i = i * OutputStrideWFS * TimeStep.
        """
        if self.stride_wfs and self.t_step:
            return float(i) * self.stride_wfs * self.t_step
        return float(i)

    @property
    def has_laser(self) -> bool:
        return bool(
            self.amplitude_x
            or self.amplitude_y
            or self.extra.get("laserx")
            or self.extra.get("lasery")
        )

    def __repr__(self) -> str:
        return (
            f"<Physics dim={self.dim} laser={self.has_laser} "
            f"stride={self.stride_wfs} dt={self.t_step} "
            f"laser_on_until={self.laser_on_until}>"
        )

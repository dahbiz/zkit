"""Automatic physics extraction from a TDSEZ input deck.

Parses the problem expressions (Potential, mass, laser fields, ...) straight
from the ``.inp`` deck via :class:`zkit.simulation.Run`, differentiates
symbolically with sympy to obtain the higher derivatives the Wigner/Moyal
analysis needs (V', V'', V''', V^(5), mass gradient), and exposes fast
lambdified callables.

Everything is derived from the deck -- no physics is hard-coded in the caller.
Expressions are read with TDSE-Z's rules (muParser syntax including nested
``c ? a : b``, every input scalar and ``Variables`` constant). Derivatives of V
come from symbolic differentiation of ``Potential``, so they are always
consistent with it. The laser field is the expression TDSE-Z applies on each
polarised axis (``LaserX/Y/Z``, or ``Laser``).

Example
-------
    >>> from zkit.simulation import Run
    >>> run = Run("build", "ZnSe_d20_A0.005338")
    >>> phys = run.physics
    >>> phys.V(0.0, 0.0)  # potential at the origin
    >>> phys.Ex(1000.0)  # laser field E_x at t=1000 a.u. (LaserX)
    >>> phys.snapshot_time(10)  # real time of wfs snapshot #10 (0-based)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import sympy as sp

__all__ = ["Physics", "parse_expression"]


# ---------------------------------------------------------------------------
# Expression parsing (muParser syntax, as used by TDSE-Z)
# ---------------------------------------------------------------------------
_SYMPY_NS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda a: sp.log(a, 10),
    "log2": lambda a: sp.log(a, 2),
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "sign": sp.sign,
    "min": sp.Min,
    "max": sp.Max,
    "Piecewise": sp.Piecewise,
}


def _top_level(s: str, ch: str, start: int = 0) -> int:
    """Index of the first ``ch`` at parenthesis depth 0 at or after ``start``."""
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ch and depth == 0:
            return i
    return -1


def _ternary_to_piecewise(s: str) -> str:
    """Rewrite muParser ``c ? a : b`` (lowest precedence, right associative,
    nested at any depth) as ``Piecewise((a, c), (b, True))``."""
    q = _top_level(s, "?")
    if q < 0:
        # no top-level ternary: convert inside every parenthesised group
        out, i = [], 0
        while i < len(s):
            if s[i] == "(":
                depth, j = 1, i + 1
                while j < len(s) and depth:
                    depth += {"(": 1, ")": -1}.get(s[j], 0)
                    j += 1
                out.append("(" + _ternary_to_piecewise(s[i + 1 : j - 1]) + ")")
                i = j
            else:
                out.append(s[i])
                i += 1
        return "".join(out)
    # matching ':' for this '?': skip the ':' of nested ternaries in the middle
    depth, nested, colon = 0, 0, -1
    for i in range(q + 1, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and c == "?":
            nested += 1
        elif depth == 0 and c == ":":
            if nested == 0:
                colon = i
                break
            nested -= 1
    if colon < 0:
        raise ValueError(f"unbalanced ternary in expression: {s!r}")
    cond = _ternary_to_piecewise(s[:q].strip())
    a = _ternary_to_piecewise(s[q + 1 : colon].strip())
    b = _ternary_to_piecewise(s[colon + 1 :].strip())
    return f"Piecewise(({a}, {cond}), ({b}, True))"


def _preprocess(src: str) -> str:
    """Normalise muParser syntax to something sympy can parse."""
    s = src.strip().replace("fabs(", "abs(")
    s = s.replace("^", "**")  # muParser power
    s = s.replace("&&", "&").replace("||", "|")
    return _ternary_to_piecewise(s)


def parse_expression(src: str, variables=("x", "y", "t"), constants=None):
    """Parse a TDSE-Z (muParser) expression into a sympy expression.

    ``constants`` (name -> value) are substituted, so the result depends only
    on the named ``variables``; see :func:`zkit.deck.expression_constants`.
    """
    s = _preprocess(src)
    syms = {v: sp.Symbol(v) for v in variables}
    consts = {k: sp.Float(v) for k, v in (constants or {}).items() if k not in syms}
    return sp.sympify(s, locals={**_SYMPY_NS, **consts, **syms})


def _axis(axis) -> str:
    return axis if isinstance(axis, str) else "xyz"[int(axis)]


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

        # ---- constants bound in TDSE-Z expressions (deck values / defaults) ----
        from zkit.deck import expression_constants

        self.constants = expression_constants(
            self.extra, dict(getattr(meta, "variables", {}) or {})
        )
        self.polarization = self.extra.get("polarization", "x")

        def _expr(key, variables, default):
            src = self.extra.get(key)
            return parse_expression(src, variables, self.constants) if src else default

        # ---- carrier: H = K + s (V + r.E), s = -Charge (electron s = +1) ----
        self.charge = self.constants["Charge"]
        self.carrier_sign = -self.charge

        # ---- potential energy of the carrier s V(x,y), and mass m(x,y) ----
        self._V_expr = self.carrier_sign * _expr("potential", ("x", "y"), sp.Integer(0))
        self._M_expr = _expr("mass", ("x", "y"), sp.Integer(1))

        # ---- laser field E(t) per axis, exactly as TDSE-Z applies it ----
        # LaserX/Y/Z, or Laser for an axis without its own expression; only
        # the axes in Polarization are coupled (the others carry no field).
        pol = "xyz" if self.polarization == "all" else self.polarization
        laser = self.extra.get("laser")
        self._E_expr = {}
        for ax in "xyz":
            src = self.extra.get("laser" + ax) or laser
            on = ax in pol and src
            self._E_expr[ax] = (
                parse_expression(src, ("t",), self.constants) if on else sp.Integer(0)
            )

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

        self._E = {ax: sp.lambdify((t,), e, "numpy") for ax, e in self._E_expr.items()}
        self._dE = {}

        # time after which the field is zero (None without a field)
        self.laser_on_until = self._detect_laser_end()

    # ------------------------------------------------------------------
    def _detect_laser_end(self) -> Optional[float]:
        """Last time in [0, FinalTime] at which the field is non-zero, found by
        evaluating E(t) on a fine grid (works for any envelope expression)."""
        if not self.has_laser:
            return None
        import numpy as np

        T = self.final_time if self.final_time > 0 else 1.0
        n = int(min(2_000_000, max(20_001, 50 * T / max(self.t_step, 1e-12))))
        tt = np.linspace(0.0, T, n)
        mag = np.zeros_like(tt)
        for f in self._E.values():
            mag = np.maximum(mag, np.abs(np.broadcast_to(f(tt), tt.shape)))
        peak = mag.max()
        if peak == 0.0:
            return None
        on = np.nonzero(mag > 1e-12 * peak)[0]
        return float(tt[min(on[-1] + 1, n - 1)])

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

    def Ex(self, t):
        """Laser field E_x(t) (LaserX); it enters H as s E_x x, s = carrier_sign."""
        return self._E["x"](t)

    def Ey(self, t):
        """Laser field E_y(t)."""
        return self._E["y"](t)

    def Ez(self, t):
        """Laser field E_z(t)."""
        return self._E["z"](t)

    def field(self, axis, t):
        """Field E_a(t) on axis 0, 1, 2 (or "x", "y", "z")."""
        return self._E[_axis(axis)](t)

    def field_expr(self, axis):
        """Field E_a(t) as a sympy expression in t (0 on an uncoupled axis)."""
        return self._E_expr[_axis(axis)]

    def field_derivative(self, axis, t):
        """dE_a/dt, differentiated symbolically from the deck's expression."""
        ax = _axis(axis)
        if ax not in self._dE:
            self._dE[ax] = sp.lambdify((self._t,), sp.diff(self._E_expr[ax], self._t), "numpy")
        return np.broadcast_to(self._dE[ax](t), np.shape(t))

    # Historical names: these return the field E(t), not a vector potential.
    Ax = Ex
    Ay = Ey

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
        return any(e != 0 for e in self._E_expr.values())

    def __repr__(self) -> str:
        return (
            f"<Physics dim={self.dim} laser={self.has_laser} "
            f"stride={self.stride_wfs} dt={self.t_step} "
            f"laser_on_until={self.laser_on_until}>"
        )

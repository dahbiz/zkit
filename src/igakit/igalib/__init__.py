"""Pure-Python subset of igakit's ``igalib.bsp`` evaluator API."""

from __future__ import annotations

import numpy as np


def _basis_matrix(degree: int, knots: np.ndarray, points: np.ndarray) -> np.ndarray:
    knots = np.asarray(knots, dtype=float)
    points = np.asarray(points, dtype=float)
    nfuncs = knots.size - degree - 1
    if degree < 1 or nfuncs < 1:
        raise ValueError("invalid B-spline degree or knot vector")
    result = np.zeros((points.size, nfuncs), dtype=float)
    for row, point in enumerate(points):
        if point < knots[degree] or point > knots[nfuncs]:
            continue
        if point == knots[nfuncs]:
            result[row, -1] = 1.0
            continue
        active = ((knots[:-1] <= point) & (point < knots[1:])).astype(float)
        for order in range(1, degree + 1):
            count = nfuncs + degree - order
            previous = active.copy()
            active = np.zeros_like(active)
            left_den = knots[order : order + count] - knots[:count]
            right_den = knots[order + 1 : order + 1 + count] - knots[1 : count + 1]
            left = np.divide(
                point - knots[:count], left_den, out=np.zeros(count), where=left_den != 0
            )
            right = np.divide(
                knots[order + 1 : order + 1 + count] - point,
                right_den,
                out=np.zeros(count),
                where=right_den != 0,
            )
            active[:count] = left * previous[:count] + right * previous[1 : count + 1]
        result[row] = active[:nfuncs]
    return result


def Evaluate1(degree, knots, control_points, points):
    return np.einsum("xi,ic->xc", _basis_matrix(degree, knots, points), control_points)


def Evaluate2(degree_x, knots_x, degree_y, knots_y, control_points, points_x, points_y):
    return np.einsum(
        "xi,yj,ijc->xyc",
        _basis_matrix(degree_x, knots_x, points_x),
        _basis_matrix(degree_y, knots_y, points_y),
        np.asarray(control_points, dtype=float),
    )


def Evaluate3(
    degree_x,
    knots_x,
    degree_y,
    knots_y,
    degree_z,
    knots_z,
    control_points,
    points_x,
    points_y,
    points_z,
):
    return np.einsum(
        "xi,yj,zk,ijkc->xyzc",
        _basis_matrix(degree_x, knots_x, points_x),
        _basis_matrix(degree_y, knots_y, points_y),
        _basis_matrix(degree_z, knots_z, points_z),
        np.asarray(control_points, dtype=float),
    )


class _BSP:
    Evaluate1 = staticmethod(Evaluate1)
    Evaluate2 = staticmethod(Evaluate2)
    Evaluate3 = staticmethod(Evaluate3)


bsp = _BSP()

__all__ = ["bsp", "Evaluate1", "Evaluate2", "Evaluate3"]

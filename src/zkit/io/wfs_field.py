"""Reconstruct TDSE-Z wavefunctions from their B-spline coefficients.

TDSE-Z stores the wavefunction as B-spline *coefficients* (the IGA degrees of
freedom), not as values on a grid.  The time-dependent snapshots are written
to ``td/wfs_<input>.h5`` with the PETSc HDF5 viewer::

    /wavefunction   (n_snap, n_x[, n_y[, n_z]], 2)   trailing axis = [Re, Im]
    /knots_x        (1-D) PetIGA compact knot vector of the x axis
    /knots_y        (1-D) same for y (2-D and 3-D runs)
    /knots_z        (1-D) same for z (3-D runs)

Each ``knots_*`` dataset carries the integer attributes ``SplineDegree`` (the
degree ``p``) and ``nfuncs`` (the number of basis functions on that axis),
which are the attributes this module reads.

The coefficients are stored in tensor ``(i, j, k)`` index order, so the
per-axis shape of a snapshot equals ``nfuncs`` and the coefficients are the
control points of a tensor-product B-spline.

The *open* knot vector expected by igakit is obtained by appending one copy
of the last knot to PetIGA's compact vector.  The field is then evaluated
with igakit's compiled evaluators ``igakit.igalib.bsp.Evaluate1``,
``Evaluate2`` and ``Evaluate3``.
"""

from __future__ import annotations

import numpy as np


def _open_knots(compact_knots, last_knot):
    """Convert a PetIGA compact knot vector to an open one (append the last knot)."""
    kv = np.asarray(compact_knots, dtype=float)
    return np.append(kv, float(last_knot))


def _reshape_static_coefficients(coeffs, nfuncs, state_name):
    """Normalize a TDSE-Z static state to ``(*nfuncs, 2)`` tensor order."""
    expected_shape = (*nfuncs, 2)
    flat_shape = (int(np.prod(nfuncs)), 2)
    if coeffs.shape == expected_shape:
        return coeffs
    if coeffs.shape == flat_shape:
        return coeffs.reshape(expected_shape)
    raise ValueError(
        f"{state_name} shape {coeffs.shape} is incompatible with the knot-derived "
        f"coefficient shape {expected_shape}"
    )


def reconstruct_wfs(path, step=0, npoints=200, axes=None):
    r"""Evaluate one wavefunction snapshot of ``td/wfs_<input>.h5`` on a grid.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``td/wfs_<input>.h5``.
    step : int, optional
        Snapshot index along the leading (time) axis of ``wavefunction``.
    npoints : int, optional
        Number of grid points per axis when ``axes`` is not given.  The
        default grid spans the whole knot span of each axis.
    axes : list of array_like, optional
        Explicit 1-D coordinate grids, one per spatial axis.  Each must lie
        inside the knot span ``[open_kv[p], open_kv[nfuncs]]`` of its axis.

    Returns
    -------
    dict
        ``dim`` (int, spatial dimension), ``step`` (int), ``axes`` (list of
        1-D coordinate arrays), ``psi`` (complex array on the tensor grid),
        ``Re`` and ``Im`` (real and imaginary parts), ``abs2`` (the density
        :math:`\lvert\psi\rvert^2`), ``SplineDegree`` and ``nfuncs`` (lists,
        one entry per axis) and ``knots`` (the open knot vectors).

    Raises
    ------
    KeyError
        If the file has no ``wavefunction`` dataset or a ``knots_*`` dataset
        is missing.
    ValueError
        If the dimensionality is not 1, 2 or 3, the coefficient shape does
        not match the ``nfuncs`` attributes, or a grid in ``axes`` lies
        outside the knot span.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "wavefunction" not in f:
            keys = list(f.keys())
            raise KeyError(
                f"{path}: no 'wavefunction' dataset/group. "
                f"Available keys: {keys}. This file is not a TDSEZ wavefunction "
                f"snapshot in the supported format (expected 'wavefunction' plus "
                f"'knots_x'[_y,_z'] with SplineDegree/nfuncs attributes)."
            )
        wave_dset = f["wavefunction"]
        coeffs = np.asarray(wave_dset[step], dtype=float)  # (..., 2)
        spatial_dim = coeffs.ndim - 1  # trailing 2 = Re/Im
        if spatial_dim not in (1, 2, 3):
            raise ValueError(
                f"unsupported wavefunction dimensionality {spatial_dim} "
                f"(dataset shape {wave_dset.shape}; expected (n_steps, n_dof_x[, n_dof_y[, "
                f"n_dof_z]], 2))"
            )
        axis_names = ("x", "y", "z")[:spatial_dim]
        compact_knots, degrees, nfuncs = [], [], []
        for axis_name in axis_names:
            key = f"knots_{axis_name}"
            if key not in f:
                raise KeyError(
                    f"{path}: missing '{key}'. The wavefunction coefficients "
                    f"cannot be reconstructed to a spatial grid without the knot "
                    f"vectors. This looks like an older/flat wfs format (coefficients "
                    f"stored as a flat dof vector without embedded knots) which is "
                    f"not supported -- re-run TDSEZ with the current binary, which "
                    f"emits tensor-ordered coefficients plus 'knots_*'."
                )
            knot_dset = f[key]
            compact_knots.append(np.asarray(knot_dset[:], dtype=float))
            degrees.append(int(knot_dset.attrs["SplineDegree"]))
            nfuncs.append(int(knot_dset.attrs["nfuncs"]))

        # --- size-compatibility guard -------------------------------------
        # The per-axis sizes of the coefficient tensor MUST equal the nfuncs
        # read from the knot vectors; otherwise the reshape below is wrong
        # (silent garbage / truncation). Fail loudly on any drift.
        coeff_axes_shape = coeffs.shape[:spatial_dim]  # drop trailing (2,) for Re/Im
        if tuple(coeff_axes_shape) != tuple(nfuncs):
            raise ValueError(
                f"{path}: wavefunction coefficient shape {coeffs.shape} (per-axis "
                f"{tuple(coeff_axes_shape)}) does not match the knot-derived basis sizes "
                f"{tuple(nfuncs)} (from knots_x/y/z nfuncs). The file is "
                f"inconsistent or uses an unsupported layout."
            )

    # homogeneous control points: (..., 3) = [weight=1, Re, Im]
    ctrl = np.zeros((*nfuncs, 3), dtype=float)
    ctrl[..., 0] = 1.0
    ctrl[..., 1] = coeffs[..., 0]
    ctrl[..., 2] = coeffs[..., 1]

    # open knot vectors + parameter grids.
    # The valid evaluation span is [open_kv[p], open_kv[nfuncs]] (i.e. the
    # open knot vector excluding the p-fold clamped end knots).  We must use
    # the OPEN vector here, not the compact one: for a single-element axis the
    # compact vector's interior collapses (kv[p] == kv[-p-1]) and yields a
    # zero-width grid.
    open_knots = [_open_knots(kv, kv[-1]) for kv in compact_knots]
    if axes is None:
        axes = [
            np.linspace(open_kv[degree], open_kv[nfunc], npoints)
            for open_kv, degree, nfunc in zip(open_knots, degrees, nfuncs)
        ]
    else:
        axes = [np.asarray(a, dtype=float) for a in axes]
        for open_kv, degree, nfunc, ax in zip(open_knots, degrees, nfuncs, axes):
            if ax.min() < open_kv[degree] - 1e-9 or ax.max() > open_kv[nfunc] + 1e-9:
                raise ValueError(
                    f"axis grid outside knot span [open_kv[p], open_kv[nfuncs]] = "
                    f"[{open_kv[degree]}, {open_kv[nfunc]}]"
                )

    psi, re, im = _evaluate_on_grid(ctrl, open_knots, degrees, nfuncs, axes)
    return {
        "dim": spatial_dim,
        "step": step,
        "axes": axes,
        "psi": psi,
        "Re": re,
        "Im": im,
        "abs2": np.abs(psi) ** 2,
        "SplineDegree": degrees,
        "nfuncs": nfuncs,
        "knots": open_knots,
    }


def _evaluate_on_grid(ctrl, open_knots, degrees, nfuncs, axes):
    """Evaluate the B-spline field on a tensor-product grid via igakit.

    ``ctrl`` is the homogeneous control-point array of shape ``(..., 3)``
    holding ``[w, Re, Im]``; ``open_knots`` are the open knot vectors,
    ``degrees`` the spline degrees, ``nfuncs`` the per-axis basis counts and
    ``axes`` the 1-D coordinate arrays.  Returns ``(psi, Re, Im)`` on the
    tensor grid (``psi`` complex, ``Re`` and ``Im`` real).
    """
    from igakit.igalib import bsp

    ctrl_c = np.ascontiguousarray(ctrl)
    knots_x = np.ascontiguousarray(open_knots[0])
    pts_x = np.ascontiguousarray(axes[0])
    spatial_dim = len(axes)
    if spatial_dim == 1:
        eval_field = bsp.Evaluate1(int(degrees[0]), knots_x, ctrl_c, pts_x)
    elif spatial_dim == 2:
        knots_y = np.ascontiguousarray(open_knots[1])
        pts_y = np.ascontiguousarray(axes[1])
        eval_field = bsp.Evaluate2(
            int(degrees[0]), knots_x, int(degrees[1]), knots_y, ctrl_c, pts_x, pts_y
        )
    else:
        knots_y = np.ascontiguousarray(open_knots[1])
        knots_z = np.ascontiguousarray(open_knots[2])
        pts_y = np.ascontiguousarray(axes[1])
        pts_z = np.ascontiguousarray(axes[2])
        eval_field = bsp.Evaluate3(
            int(degrees[0]),
            knots_x,
            int(degrees[1]),
            knots_y,
            int(degrees[2]),
            knots_z,
            ctrl_c,
            pts_x,
            pts_y,
            pts_z,
        )

    # eval_field[..., 0] = weight w ; eval_field[..., 1:] = (Re, Im) * w  per point
    w = eval_field[..., 0]
    re = eval_field[..., 1] / w
    im = eval_field[..., 2] / w
    psi = re + 1j * im
    return psi, re, im


def _gauss_grid(open_knots, degree, nq=5):
    """One-axis Gauss-Legendre rule over the non-empty knot spans.

    The ``nq``-point Legendre rule on ``[-1, 1]`` is mapped into every span
    between consecutive distinct knots of ``open_knots[p:nfuncs + 1]``, with
    the Jacobian (half the span length) folded into the weights, so the rule
    also applies to non-uniform knot vectors.  Returns ``(points, weights)``
    as 1-D arrays.
    """
    nfuncs = len(open_knots) - degree - 1
    # unique interior knot values (the spline breaks), excluding the
    # p-fold clamped ends: U[p : nfuncs+1]
    breaks = []
    seen = set()
    for value in open_knots[degree : nfuncs + 1]:
        if value not in seen:
            seen.add(value)
            breaks.append(value)
    breaks = np.asarray(breaks)
    gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(nq)
    all_pts, all_wts = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        half_span = 0.5 * (b - a)  # half span (Jacobian of the map)
        center = 0.5 * (a + b)
        all_pts.append(center + half_span * gauss_pts)
        all_wts.append(half_span * gauss_wts)
    return np.concatenate(all_pts), np.concatenate(all_wts)


def compute_wfs_norm(path, step=0, nq=5):
    r"""Norm of a wavefunction snapshot by Gauss quadrature over the knot spans.

    Computes :math:`\langle\psi\vert\psi\rangle = \int \lvert\psi\rvert^2\,d^dr`
    over the whole spline domain with a tensor-product Gauss-Legendre rule
    of ``nq`` points per knot span on each axis.  Within a span the density
    is a polynomial of degree ``2p`` per axis, so the rule is exact (up to
    round-off) whenever ``2p <= 2*nq - 1``; the default ``nq=5`` is exact for
    splines of degree ``p <= 4``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``td/wfs_<input>.h5``.
    step : int, optional
        Snapshot index.
    nq : int, optional
        Gauss points per knot span and axis.

    Returns
    -------
    float
        The integral of :math:`\lvert\psi\rvert^2` over the domain.

    Raises
    ------
    KeyError
        If ``wavefunction`` or a ``knots_*`` dataset is missing.
    ValueError
        If the coefficient shape does not match the ``nfuncs`` attributes.
    """
    import h5py

    from igakit.igalib import bsp

    with h5py.File(path, "r") as f:
        if "wavefunction" not in f:
            raise KeyError(f"{path}: no 'wavefunction' dataset/group")
        wave_dset = f["wavefunction"]
        coeffs = np.asarray(wave_dset[step], dtype=float)
        spatial_dim = coeffs.ndim - 1
        axis_names = ("x", "y", "z")[:spatial_dim]
        degrees, nfuncs, open_knots = [], [], []
        for axis_name in axis_names:
            key = f"knots_{axis_name}"
            if key not in f:
                raise KeyError(f"{path}: missing '{key}' (knot vectors required)")
            knot_dset = f[key]
            degrees.append(int(knot_dset.attrs["SplineDegree"]))
            nfuncs.append(int(knot_dset.attrs["nfuncs"]))
            kv = np.asarray(knot_dset[:], dtype=float)
            open_knots.append(np.append(kv, kv[-1]))
        # size-compatibility guard (same check as in reconstruct_wfs)
        if tuple(coeffs.shape[:spatial_dim]) != tuple(nfuncs):
            raise ValueError(
                f"{path}: coefficient shape {coeffs.shape} (per-axis "
                f"{tuple(coeffs.shape[:spatial_dim])}) != knot-derived nfuncs "
                f"{tuple(nfuncs)}"
            )

    ctrl = np.zeros((*nfuncs, 3), dtype=float)
    ctrl[..., 0] = 1.0
    ctrl[..., 1] = coeffs[..., 0]
    ctrl[..., 2] = coeffs[..., 1]
    ctrl_c = np.ascontiguousarray(ctrl)

    gauss_grids = [_gauss_grid(open_kv, degree, nq) for open_kv, degree in zip(open_knots, degrees)]
    quad_points = [g[0] for g in gauss_grids]
    quad_weights = [g[1] for g in gauss_grids]
    pts_x = np.ascontiguousarray(quad_points[0])
    if spatial_dim == 1:
        eval_field = bsp.Evaluate1(
            int(degrees[0]), np.ascontiguousarray(open_knots[0]), ctrl_c, pts_x
        )
        tensor_weights = quad_weights[0]
    elif spatial_dim == 2:
        pts_y = np.ascontiguousarray(quad_points[1])
        eval_field = bsp.Evaluate2(
            int(degrees[0]),
            np.ascontiguousarray(open_knots[0]),
            int(degrees[1]),
            np.ascontiguousarray(open_knots[1]),
            ctrl_c,
            pts_x,
            pts_y,
        )
        tensor_weights = np.outer(quad_weights[0], quad_weights[1])
    else:
        pts_y = np.ascontiguousarray(quad_points[1])
        pts_z = np.ascontiguousarray(quad_points[2])
        eval_field = bsp.Evaluate3(
            int(degrees[0]),
            np.ascontiguousarray(open_knots[0]),
            int(degrees[1]),
            np.ascontiguousarray(open_knots[1]),
            int(degrees[2]),
            np.ascontiguousarray(open_knots[2]),
            ctrl_c,
            pts_x,
            pts_y,
            pts_z,
        )
        tensor_weights = np.multiply.outer(
            np.multiply.outer(quad_weights[0], quad_weights[1]), quad_weights[2]
        )
    psi = (eval_field[..., 1] + 1j * eval_field[..., 2]) / eval_field[..., 0]
    return float(np.sum(np.abs(psi) ** 2 * tensor_weights))


def wfs_to_vtk(res, out_path, fmt=None):
    """Export an evaluated wavefunction to a VTK file for ParaView.

    Parameters
    ----------
    res : dict
        Output of :func:`reconstruct_wfs` or :func:`reconstruct_static_wfs`.
    out_path : str or pathlib.Path
        Destination file.  Unless ``fmt`` is given, the extension selects
        the format: ``.vts`` for a VTK structured grid, ``.vtu`` for an
        unstructured grid (the structured grid cast to unstructured) and
        ``.vtk`` for the legacy VTK format.  Any other extension is written
        as a structured grid.
    fmt : {"vts", "vtu", "vtk"}, optional
        Explicit format, overriding the extension.

    Returns
    -------
    str
        ``out_path``.

    Notes
    -----
    The grid carries the point-data arrays ``psi_re`` (Re psi), ``psi_im``
    (Im psi), ``psi_mag`` (the density |psi|^2, taken from ``res["abs2"]``)
    and ``psi_phase`` (arg psi, in radians).  1-D and 2-D fields are written
    as structured grids with singleton extents in the missing directions.

    Requires ``pyvista`` (the ``viz`` extra).
    """
    import pyvista as pv

    out_path = str(out_path)
    if fmt is None:
        ext = out_path.rsplit(".", 1)[-1].lower()
        fmt = {"vts": "vts", "vtu": "vtu", "vtk": "vtk"}.get(ext, "vts")
    fmt = fmt.lower()

    spatial_dim = res["dim"]
    axes = res["axes"]
    re = np.asarray(res["Re"]).ravel()
    im = np.asarray(res["Im"]).ravel()
    mag = np.asarray(res["abs2"]).ravel()
    phase = np.angle(np.asarray(res["psi"])).ravel()

    if spatial_dim == 1:
        x = np.asarray(axes[0], dtype=float)
        pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = [x.size, 1, 1]
    elif spatial_dim == 2:
        X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = [axes[0].size, axes[1].size, 1]
    else:  # 3D
        X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = [axes[0].size, axes[1].size, axes[2].size]

    grid.point_data["psi_re"] = re
    grid.point_data["psi_im"] = im
    grid.point_data["psi_mag"] = mag
    grid.point_data["psi_phase"] = phase

    if fmt == "vtu":
        # convert the structured grid to an unstructured one (still a grid)
        grid = grid.cast_to_unstructured_grid()

    grid.save(out_path)
    return out_path


def reconstruct_static_wfs(path, istate=0, npoints=200, axes=None):
    """Evaluate the static eigenstate ``psi_<istate>`` of an EigenData file on a grid.

    The static solver of TDSE-Z stores each converged eigenstate as a
    top-level dataset ``psi_<i>`` of B-spline coefficients with a trailing
    ``[Re, Im]`` axis, either in tensor shape ``(*nfuncs, 2)`` or flattened
    to ``(prod(nfuncs), 2)`` in tensor order, together with top-level
    ``knots_x`` (and ``knots_y``, ``knots_z``) datasets carrying the
    ``SplineDegree`` and ``nfuncs`` attributes.  The spatial dimension is the
    number of ``knots_*`` datasets present.  Evaluation follows
    :func:`reconstruct_wfs`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``static/EigenData_<input>.h5``.
    istate : int, optional
        Index of the eigenstate (the ``<i>`` in ``psi_<i>``).
    npoints : int, optional
        Grid points per axis when ``axes`` is not given.
    axes : list of array_like, optional
        Explicit 1-D coordinate grids, one per spatial axis, each inside the
        knot span ``[open_kv[p], open_kv[nfuncs]]``.

    Returns
    -------
    dict
        The same keys as :func:`reconstruct_wfs`, with ``istate`` in place
        of ``step``.

    Raises
    ------
    KeyError
        If the dataset ``psi_<istate>`` does not exist.
    ValueError
        If the coefficient shape is incompatible with ``nfuncs`` or a grid
        lies outside the knot span.
    """
    import h5py

    state_name = f"psi_{istate}"
    with h5py.File(path, "r") as f:
        if state_name not in f:
            avail = [k for k in f.keys() if k.startswith("psi_")]
            raise KeyError(f"{path}: no dataset '{state_name}'. Available states: {avail}")
        coeffs = np.asarray(f[state_name], dtype=float)  # (n_dof, 2) = [Re, Im]
        axis_names = tuple(k for k in ("x", "y", "z") if f"knots_{k}" in f)
        spatial_dim = len(axis_names)
        compact_knots, degrees, nfuncs = [], [], []
        for axis_name in axis_names:
            knot_dset = f[f"knots_{axis_name}"]
            compact_knots.append(np.asarray(knot_dset[:], dtype=float))
            degrees.append(int(knot_dset.attrs["SplineDegree"]))
            nfuncs.append(int(knot_dset.attrs["nfuncs"]))

        coeffs = _reshape_static_coefficients(coeffs, nfuncs, f"{path}: {state_name}")

    ctrl = np.zeros((*nfuncs, 3), dtype=float)
    ctrl[..., 0] = 1.0
    ctrl[..., 1] = coeffs[..., 0]
    ctrl[..., 2] = coeffs[..., 1]

    open_knots = [_open_knots(kv, kv[-1]) for kv in compact_knots]
    if axes is None:
        axes = [
            np.linspace(open_kv[degree], open_kv[nfunc], npoints)
            for open_kv, degree, nfunc in zip(open_knots, degrees, nfuncs)
        ]
    else:
        axes = [np.asarray(a, dtype=float) for a in axes]
        for open_kv, degree, nfunc, ax in zip(open_knots, degrees, nfuncs, axes):
            if ax.min() < open_kv[degree] - 1e-9 or ax.max() > open_kv[nfunc] + 1e-9:
                raise ValueError(
                    f"axis grid outside knot span [open_kv[p], open_kv[nfuncs]] "
                    f"= [{open_kv[degree]}, {open_kv[nfunc]}]"
                )

    psi, re, im = _evaluate_on_grid(ctrl, open_knots, degrees, nfuncs, axes)
    return {
        "dim": spatial_dim,
        "istate": istate,
        "axes": axes,
        "psi": psi,
        "Re": re,
        "Im": im,
        "abs2": np.abs(psi) ** 2,
        "SplineDegree": degrees,
        "nfuncs": nfuncs,
        "knots": open_knots,
    }


__all__ = [
    "reconstruct_wfs",
    "reconstruct_static_wfs",
    "wfs_to_vtk",
    "compute_wfs_norm",
]

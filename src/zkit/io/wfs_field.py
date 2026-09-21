"""Reconstruct TDSEZ wavefunctions from B-spline coefficients using igakit.

TDSEZ writes the solution psi as B-spline *coefficients* (the IGA dof vector)
into ``td/wfs_<input>.h5`` via the PETSc HDF5 viewer.  The file layout is:

    /wavefunction                  (n_snap, n_dof_x[, n_dof_y[, n_dof_z]], 2)  [Re, Im]
    /knots_x, /knots_y, /knots_z   (1-D)  PetIGA *compact* knot vector
    /knots_x.attrs:
        SplineDegree  (int)
        nfuncs       (int)   number of basis functions on that axis
        LMin, LMax   (float) domain bounds

The coefficients are stored in tensor (i, j, k) index order (NOT the PetIGA
node-closure global dof order), so a simple reshape to ``(nx, ny, nz, 2)``
recovers the control points.

We rebuild the *open* knot vector PetIGA/igakit expect by appending one
trailing copy of the last knot (PetIGA's compact vector already has the
right-end knot at multiplicity ``p``, so only ONE extra copy is needed to
reach the open multiplicity ``p+1``).  Then we evaluate the B-spline with
igakit's compiled evaluator ``igakit.igalib.bsp.Evaluate{1,2,3}`` (the
high-level ``NURBS.evaluate()`` is broken in igakit 0.1.0 -- it returns an
empty array -- but the compiled engine works perfectly).
"""

from __future__ import annotations

import numpy as np


def _open_knots(compact_knots, last_knot):
    """PetIGA compact vector -> open vector (append one trailing copy)."""
    kv = np.asarray(compact_knots, dtype=float)
    return np.append(kv, float(last_knot))


def _reshape_static_coefficients(coeffs, nfuncs, state_name):
    """Normalize a TDSEZ static state to ``(*nfuncs, 2)`` tensor order."""
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
    """Reconstruct the wavefunction on a spatial grid from td/wfs_*.h5.

    Parameters
    ----------
    path : str | Path
        Path to ``td/wfs_<input>.h5``.
    step : int
        Snapshot index into the leading (time) axis.
    npoints : int
        Number of grid points per axis when ``axes`` is not given.
    axes : list[array-like] | None
        Optional explicit coordinate grids (one per spatial axis).  Each must
        lie inside the knot span ``[kv[p], kv[-p-1]]``.

    Returns
    -------
    dict with keys:
        dim, step, axes, psi (complex), Re, Im, abs2,
        SplineDegree, nfuncs, knots (open knot vectors).
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

    ``ctrl`` is the homogeneous control-point array (..., 3) = [w, Re, Im].
    ``open_knots`` are the open (clamped) knot vectors, ``degrees`` the spline
    degrees, ``nfuncs`` the per-axis basis counts, ``axes`` the 1-D coordinate
    arrays. Returns ``(psi, Re, Im)`` complex arrays on the tensor grid.
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
    """Tensor-product Gauss quadrature rule over the knot-vector spans.

    Builds the Gauss points and weights for one axis by mapping the reference
    Legendre rule on [-1, 1] into every *interior* knot span of the open knot
    vector (igakit's ``EvalBasisFuns``/``FindSpan`` operate on the open vector).
    The per-span Jacobian (span length) is folded into the weights, so this
    integrates exactly on *non-uniform* knot vectors too.

    Returns ``(points, weights)`` 1-D arrays.
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
    """Exact <psi|psi> via Gauss quadrature over the knot spans (igakit).

    Reads the wavefunction + knot vectors from ``path`` and integrates |psi|^2
    with a tensor-product Gauss rule mapped into every interior knot span, using
    igakit's ``Evaluate{1,2,3}`` basis engine.  The per-span Jacobian (span
    length) is folded into the weights, so the result is exact to the chosen
    Gauss order (nq=5 already gives ~1e-15 for smooth splines) and works for
    *non-uniform* knot vectors.  This is machine-accurate, unlike the
    rectangular rule over a uniform plot grid.

    Parameters
    ----------
    path : str | Path
        Path to ``td/wfs_<input>.h5``.
    step : int
        Snapshot index.
    nq : int
        Gauss points per knot span (default 5 -> exact for degree <= 9 splines).

    Returns
    -------
    float : the squared norm integral of |psi|^2 over the whole domain.
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
        # size-compatibility guard (mirror of evaluate_wavefunction)
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
    """Export an evaluated wavefunction (from :func:`reconstruct_wfs`)
    to a VTK file for ParaView.

    Parameters
    ----------
    res : dict
        Output of :func:`reconstruct_wfs`.
    out_path : str | Path
        Destination file.  The extension chooses the format:
        ``.vts`` -> VTK StructuredGrid (recommended, native regular grid),
        ``.vtu`` -> VTK UnstructuredGrid (polydata points + cells),
        ``.vtk`` -> legacy VTK StructuredGrid.  If ``fmt`` is given it
        overrides the extension.
    fmt : str | None
        Optional explicit format: ``"vts"``, ``"vtu"`` or ``"vtk"``.

    The grid carries point data: ``psi_re``, ``psi_im``, ``psi_mag`` (|psi|),
    and ``psi_phase`` (arg psi).  For 1D the y/z extents are collapsed to a
    single layer so ParaView still reads it as a valid structured grid.

    Requires ``pyvista`` (declared in the ``viz`` extra).
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
    """Reconstruct a static eigenstate psi_<istate> from EigenData_*.h5.

    TDSEZ's static writer stores each converged eigenstate as a top-level
    dataset ``psi_<i>`` (the IGA dof/coefficient vector, written by PETSc's
    HDF5 viewer, i.e. already in tensor/i,j,k coefficient order) plus a
    top-level ``knots_x[_y,_z]`` carrying the {SplineDegree, nfuncs,
    mult_start_end} attributes.  This is the same information the propagation
    writer emits, just split per-state instead of stacked under ``wavefunction``.

    The reconstruction reuses the exact igakit B-spline engine PetIGA uses.

    Parameters
    ----------
    path : str | Path
        Path to ``static/EigenData_<input>.h5``.
    istate : int
        Index of the eigenstate (the ``<i>`` in ``psi_<i>``).
    npoints : int
        Grid points per axis when ``axes`` is None.
    axes : list[array-like] | None
        Optional explicit coordinate grids (one per spatial axis), each inside
        the knot span ``[open_kv[p], open_kv[nfuncs]]``.

    Returns
    -------
    dict (same keys as :func:`reconstruct_wfs`):
        dim, istate, axes, psi, Re, Im, abs2, SplineDegree, nfuncs, knots.
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

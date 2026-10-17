"""Export TDSE-Z runs to VTK for ParaView and VisIt (no VTK library needed).

Every file is VTK XML ImageData (``.vti``, uniform grid, base64 binary,
Float32), which ParaView and VisIt both read. A time series also gets

* ``<name>.pvd``: ParaView collection with the physical times, and
* ``<name>.visit``: VisIt file list (the time is in each file as the
  ``TIME``/``CYCLE`` field data, which VisIt's VTK reader picks up).

Arrays (point data, ``fields``): by default ``density`` |psi|^2,
``log10_density`` (floored at ``floor`` times the maximum of the snapshot) and,
when the deck's potential can be evaluated, ``potential`` (s V, the energy of
the carrier; V_KH for ``Mode = kh``); ``fields="all"`` adds ``re``, ``im`` and
``phase``. Field data: ``TIME``, ``CYCLE``,
``TimeValue`` and the laser ``E_x``, ``E_y``, ``E_z`` at that time.

Coordinates are x, y, z in a.u.; 1D and 2D runs are written with
singleton extents in the missing directions.
"""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def _b64(a: np.ndarray, compress: bool) -> str:
    """VTK XML inline binary: UInt32 header + Float32 data, base64; zlib in one block."""
    raw = np.ascontiguousarray(a, dtype="<f4").tobytes()
    if not compress:
        return base64.b64encode(struct.pack("<I", len(raw)) + raw).decode("ascii")
    comp = zlib.compress(raw, 6)
    header = struct.pack("<4I", 1, len(raw), len(raw), len(comp))
    return base64.b64encode(header).decode("ascii") + base64.b64encode(comp).decode("ascii")


def write_vti(
    path: PathLike,
    axes: Sequence[np.ndarray],
    point_data: Dict[str, np.ndarray],
    field_data: Optional[Dict[str, float]] = None,
    compress: bool = True,
) -> Path:
    """Write uniform-grid point data to a VTK XML ImageData file (zlib-compressed by default).

    ``axes`` are the 1-3 coordinate vectors (uniform spacing); each array of
    ``point_data`` has shape ``(len(axes[0]), len(axes[1]), ...)`` in (x, y, z)
    order. ``field_data`` holds scalars attached to the whole data set.
    """
    axes = [np.asarray(a, dtype=float) for a in axes]
    n = [a.size for a in axes] + [1] * (3 - len(axes))
    origin = [a[0] for a in axes] + [0.0] * (3 - len(axes))
    spacing = [(a[-1] - a[0]) / (a.size - 1) if a.size > 1 else 1.0 for a in axes] + [1.0] * (3 - len(axes))
    for a, h in zip(axes, spacing):
        if a.size > 2 and np.max(np.abs(np.diff(a) - h)) > 1e-9 * max(1.0, abs(h)):
            raise ValueError("write_vti needs uniformly spaced axes")
    extent = " ".join(f"0 {k - 1}" for k in n)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32"'
        + (' compressor="vtkZLibDataCompressor">' if compress else ">"),
        f'  <ImageData WholeExtent="{extent}" Origin="{" ".join(f"{o:.10g}" for o in origin)}" '
        f'Spacing="{" ".join(f"{h:.10g}" for h in spacing)}">',
    ]
    if field_data:
        lines.append("    <FieldData>")
        for name, v in field_data.items():
            typ = "Int32" if name == "CYCLE" else "Float64"
            lines.append(f'      <DataArray type="{typ}" Name="{name}" NumberOfTuples="1" format="ascii">{v}</DataArray>')
        lines.append("    </FieldData>")
    lines.append(f'    <Piece Extent="{extent}">')
    first = next(iter(point_data), None)
    lines.append(f'      <PointData Scalars="{first}">' if first else "      <PointData>")
    for name, arr in point_data.items():
        arr = np.asarray(arr)
        if arr.shape != tuple(a.size for a in axes):
            raise ValueError(f"{name}: shape {arr.shape} does not match the axes {tuple(a.size for a in axes)}")
        flat = arr.ravel(order="F")  # VTK: x fastest
        lines.append(f'        <DataArray type="Float32" Name="{name}" format="binary" '
                     f'RangeMin="{np.nanmin(flat):.7g}" RangeMax="{np.nanmax(flat):.7g}">')
        lines.append("          " + _b64(flat, compress))
        lines.append("        </DataArray>")
    lines += ["      </PointData>", "      <CellData/>", "    </Piece>", "  </ImageData>", "</VTKFile>", ""]
    path = Path(path)
    path.write_text("\n".join(lines))
    return path


def write_pvd(path: PathLike, files: Iterable[PathLike], times: Iterable[float]) -> Path:
    """ParaView collection of a time series (file names relative to ``path``)."""
    path = Path(path)
    rows = [f'    <DataSet timestep="{t:.10g}" part="0" file="{Path(f).name}"/>' for f, t in zip(files, times)]
    path.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0">\n  <Collection>\n'
                    + "\n".join(rows) + "\n  </Collection>\n</VTKFile>\n")
    return path


def write_visit(path: PathLike, files: Iterable[PathLike]) -> Path:
    """VisIt file list of a time series (one file per line, one block)."""
    path = Path(path)
    path.write_text("!NBLOCKS 1\n" + "\n".join(Path(f).name for f in files) + "\n")
    return path


# ---------------------------------------------------------------------------
# fast tensor-product B-spline evaluation (basis matrices built once per axis)
# ---------------------------------------------------------------------------
def _basis_matrix(knots: np.ndarray, p: int, nfunc: int, x: np.ndarray) -> np.ndarray:
    """B_j(x_i) of the open knot vector ``knots`` (Cox-de Boor), shape (len(x), nfunc)."""
    t = np.asarray(knots, dtype=float)
    x = np.clip(np.asarray(x, dtype=float), t[p], t[nfunc])
    nk = t.size
    B = ((t[None, :-1] <= x[:, None]) & (x[:, None] < t[None, 1:])).astype(float)
    # right end of the span: the last non-empty interval holds x = t[nfunc]
    last = np.nonzero(t[:-1] < t[1:])[0]
    last = last[last < nfunc][-1]
    B[x >= t[nfunc], :] = 0.0
    B[x >= t[nfunc], last] = 1.0
    for k in range(1, p + 1):
        n = nk - k - 1
        d1 = t[k:k + n] - t[:n]
        d2 = t[k + 1:k + 1 + n] - t[1:1 + n]
        a = np.divide(x[:, None] - t[None, :n], d1, out=np.zeros((x.size, n)), where=d1 > 0)
        b = np.divide(t[None, k + 1:k + 1 + n] - x[:, None], d2, out=np.zeros((x.size, n)), where=d2 > 0)
        B = a * B[:, :n] + b * B[:, 1:n + 1]
    return B[:, :nfunc]


class _Evaluator:
    """psi on a tensor grid from the B-spline coefficients of a TDSE-Z file."""

    def __init__(self, path: PathLike, npoints: int):
        import h5py

        from zkit.io.wfs_field import _open_knots

        with h5py.File(path, "r") as f:
            names = [a for a in "xyz" if f"knots_{a}" in f]
            self.p, self.nfuncs, knots = [], [], []
            for a in names:
                d = f[f"knots_{a}"]
                kv = np.asarray(d[:], dtype=float)
                knots.append(_open_knots(kv, kv[-1]))
                self.p.append(int(d.attrs["SplineDegree"]))
                self.nfuncs.append(int(d.attrs["nfuncs"]))
        self.axes = [np.linspace(kv[p], kv[n], npoints) for kv, p, n in zip(knots, self.p, self.nfuncs)]
        self.B = [_basis_matrix(kv, p, n, x) for kv, p, n, x in zip(knots, self.p, self.nfuncs, self.axes)]

    def __call__(self, coeffs: np.ndarray) -> dict:
        from zkit.io.wfs_field import _to_xyz

        c = _to_xyz(np.asarray(coeffs, dtype=float), self.nfuncs, "coefficients")
        c = c[..., 0] + 1j * c[..., 1]
        for ax, B in enumerate(self.B):  # contract axis ax with its basis matrix
            c = np.moveaxis(np.tensordot(B, c, axes=([1], [ax])), 0, ax)
        return {"psi": c, "Re": c.real, "Im": c.imag, "abs2": np.abs(c) ** 2, "axes": self.axes}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
ALL_FIELDS = ("density", "log10_density", "re", "im", "phase", "potential")
DEFAULT_FIELDS = ("density", "log10_density", "potential")


def _fields(res: dict, floor: float, fields: Sequence[str]) -> Dict[str, np.ndarray]:
    dens = np.asarray(res["abs2"], dtype=float)
    peak = float(dens.max()) if dens.size else 0.0
    make = {
        "density": lambda: dens,
        "log10_density": lambda: np.log10(np.maximum(dens, floor * peak if peak > 0 else floor)),
        "re": lambda: np.asarray(res["Re"], dtype=float),
        "im": lambda: np.asarray(res["Im"], dtype=float),
        "phase": lambda: np.angle(np.asarray(res["psi"])),
    }
    return {k: make[k]() for k in fields if k in make}


def _check_fields(fields) -> tuple:
    if fields in (None, "default"):
        return DEFAULT_FIELDS
    if fields == "all":
        return ALL_FIELDS
    fields = tuple(fields.split(",")) if isinstance(fields, str) else tuple(fields)
    bad = [f for f in fields if f not in ALL_FIELDS]
    if bad:
        raise ValueError(f"unknown fields {bad}; choose from {ALL_FIELDS}, 'default' or 'all'")
    return fields


def _potential(run, axes: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """s V (or s V_KH for Mode = kh) on the grid, if the deck's potential parses."""
    try:
        import sympy as sp

        from zkit.physics import parse_expression
    except ImportError:
        return None
    try:
        phys = run.physics
        src = phys.extra.get("potential")
        if not src:
            return None
        names = ("x", "y", "z")
        expr = parse_expression(src, names, phys.constants)
        f = sp.lambdify([sp.Symbol(v) for v in names], expr, "numpy")
        dim = len(axes)
        grids = list(np.meshgrid(*axes, indexing="ij"))
        while len(grids) < 3:
            grids.append(np.zeros_like(grids[0]))

        def V(*c):
            cc = list(c) + [np.zeros_like(c[0])] * (3 - len(c))
            return np.broadcast_to(f(*cc), c[0].shape).astype(float)

        if str(phys.extra.get("mode", "td")).strip().lower() == "kh":
            v = run.kh.average(V, *grids[:dim])
        else:
            v = V(*grids[:dim])
        return phys.carrier_sign * np.asarray(v, dtype=float)
    except Exception:  # noqa: BLE001 - optional decoration of the export
        return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def export_states(
    run,
    out_dir: PathLike,
    states: Optional[Sequence[int]] = None,
    npoints: int = 160,
    floor: float = 1e-8,
    fields=None,
) -> list:
    """Write each saved eigenstate (``NBoundStatesSave = 1``) to ``<name>_state<i>.vti``.

    Returns the written paths. The potential (or V_KH) is added to every file.
    """
    import h5py

    eig = run.path("eigen")
    with h5py.File(eig, "r") as f:
        saved = sorted(int(k[4:]) for k in f if k.startswith("psi_"))
        energies = np.asarray(f["spectrum"])[:, 0] if "spectrum" in f else None
        if not saved:
            raise ValueError(f"{eig}: no saved states (set NBoundStatesSave = 1)")
        states = saved if states is None else [s for s in states if s in saved]
        ev = _Evaluator(eig, npoints)
        coeffs = {i: np.asarray(f[f"psi_{i}"]) for i in states}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = _check_fields(fields)
    pot = _potential(run, ev.axes) if "potential" in fields else None
    written = []
    for i in states:
        data = _fields(ev(coeffs[i]), floor, fields)
        if pot is not None:
            data["potential"] = pot
        fd = {"STATE": float(i)}
        if energies is not None and i < energies.size:
            fd["ENERGY"] = float(energies[i])
        written.append(write_vti(out_dir / f"{run.name}_state{i}.vti", ev.axes, data, fd))
    return written


def export_evolution(
    run,
    out_dir: PathLike,
    every: int = 1,
    npoints: int = 160,
    floor: float = 1e-8,
    fields=None,
    first: int = 0,
    last: Optional[int] = None,
) -> Path:
    """Write the wavefunction snapshots of ``td/<name>_wfs.h5`` as a ``.vti`` series.

    Snapshot ``k`` is at t = k * OutputStrideWFS * TimeStep. Writes
    ``<name>_wfs_<k>.vti``, ``<name>_wfs.pvd`` (ParaView) and
    ``<name>_wfs.visit`` (VisIt); returns the ``.pvd`` path.
    """
    import h5py

    wfs = run.path("wfs")
    ev = _Evaluator(wfs, npoints)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run.meta
    dt, stride = float(meta.time_step or 0.0), int(meta.output_stride_wfs or 0)
    fields = _check_fields(fields)
    pot = _potential(run, ev.axes) if "potential" in fields else None
    try:
        phys = run.physics
        efield = [lambda t, a=a: float(np.asarray(phys.field(a, t))) for a in range(3)]
    except Exception:  # noqa: BLE001 - sympy missing or unparsable laser
        efield = None
    files, times = [], []
    with h5py.File(wfs, "r") as f:
        dset = f["wavefunction"]
        nsnap = int(dset.shape[0])
        last = nsnap if last is None else min(last, nsnap)
        steps = list(range(first, last, max(1, every)))
        if not steps:
            raise ValueError(f"{wfs}: no snapshots in [{first}, {last})")
        for k in steps:
            t = k * stride * dt if stride and dt else float(k)
            data = _fields(ev(dset[k]), floor, fields)
            if pot is not None:
                data["potential"] = pot
            fd = {"TIME": t, "TimeValue": t, "CYCLE": k}
            if efield:
                fd.update({f"E_{c}": efield[a](t) for a, c in enumerate("xyz")})
            files.append(write_vti(out_dir / f"{run.name}_wfs_{k:05d}.vti", ev.axes, data, fd))
            times.append(t)
    write_visit(out_dir / f"{run.name}_wfs.visit", files)
    return write_pvd(out_dir / f"{run.name}_wfs.pvd", files, times)


__all__ = ["ALL_FIELDS", "DEFAULT_FIELDS", "write_vti", "write_pvd", "write_visit", "export_states", "export_evolution"]

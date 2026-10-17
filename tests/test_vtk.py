"""VTK export for ParaView/VisIt: file format, B-spline evaluation, series files."""

import base64
import re
import struct
import subprocess
import zlib

import h5py
import numpy as np
import pytest

from zkit.viz.vtk import _basis_matrix, _Evaluator, write_pvd, write_visit, write_vti


def _decode(text, name, compressed):
    """Array ``name`` of a .vti written by write_vti, decoded independently."""
    m = re.search(rf'Name="{name}" format="binary"[^>]*>\s*(\S+)\s*<', text)
    blob = m.group(1)
    if not compressed:
        raw = base64.b64decode(blob)
        n = struct.unpack("<I", raw[:4])[0]
        return np.frombuffer(raw[4:4 + n], "<f4")
    header = base64.b64decode(blob[:24])            # 4 UInt32 -> 16 bytes -> 24 base64 chars
    nb, usize, _, csize = struct.unpack("<4I", header)
    assert nb == 1
    data = zlib.decompress(base64.b64decode(blob[24:]))
    assert len(data) == usize
    return np.frombuffer(data, "<f4")


@pytest.mark.parametrize("compress", [True, False])
def test_write_vti_round_trip_x_fastest(tmp_path, compress):
    x, y, z = np.linspace(-2, 2, 5), np.linspace(0, 3, 4), np.linspace(1, 2, 3)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    f = X + 10 * Y + 100 * Z
    p = write_vti(tmp_path / "a.vti", [x, y, z], {"f": f}, {"TIME": 1.5, "CYCLE": 3}, compress=compress)
    text = p.read_text()
    assert 'WholeExtent="0 4 0 3 0 2"' in text and 'Origin="-2 0 1"' in text
    assert 'Name="TIME" NumberOfTuples="1" format="ascii">1.5<' in text
    got = _decode(text, "f", compress)
    assert np.allclose(got, f.ravel(order="F"))      # VTK point order: x fastest
    assert got[1] - got[0] == pytest.approx(1.0)       # neighbour along x


def test_write_vti_rejects_nonuniform_and_bad_shape(tmp_path):
    with pytest.raises(ValueError, match="uniform"):
        write_vti(tmp_path / "b.vti", [np.array([0.0, 1.0, 3.0])], {"f": np.zeros(3)})
    with pytest.raises(ValueError, match="shape"):
        write_vti(tmp_path / "b.vti", [np.linspace(0, 1, 3)], {"f": np.zeros(4)})


def test_series_files(tmp_path):
    files = [tmp_path / f"s_{k}.vti" for k in range(3)]
    pvd = write_pvd(tmp_path / "s.pvd", files, [0.0, 0.5, 1.0]).read_text()
    assert pvd.count("<DataSet") == 3 and 'timestep="0.5"' in pvd and 'file="s_1.vti"' in pvd
    vis = write_visit(tmp_path / "s.visit", files).read_text().split()
    assert vis == ["!NBLOCKS", "1", "s_0.vti", "s_1.vti", "s_2.vti"]


def test_basis_matrix_partition_of_unity_and_end_points():
    p, n = 3, 9
    t = np.r_[[0.0] * p, np.linspace(0, 6, n - p + 1), [6.0] * p]   # open, n + p + 1 knots
    x = np.linspace(0, 6, 101)
    B = _basis_matrix(t, p, n, x)
    assert B.shape == (101, n)
    assert np.allclose(B.sum(1), 1.0) and B.min() >= -1e-14
    assert B[0, 0] == pytest.approx(1.0) and B[-1, -1] == pytest.approx(1.0)


def _wfs_file(path, nfuncs, degree=3):
    """Synthetic td/<name>_wfs.h5: [z][y][x] coefficients plus PetIGA compact knots."""
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        for a, n in zip("xyz", nfuncs):
            kv = np.r_[[-4.0] * degree, np.linspace(-4, 4, n - degree + 1), [4.0] * (degree - 1)]
            d = f.create_dataset(f"knots_{a}", data=kv)
            d.attrs["SplineDegree"] = degree
            d.attrs["nfuncs"] = n
        f["wavefunction"] = rng.normal(size=(2, *reversed(nfuncs), 2))


@pytest.mark.parametrize("nfuncs", [(7,), (7, 9), (6, 7, 8)])
def test_evaluator_matches_igakit_reconstruction(tmp_path, nfuncs):
    pytest.importorskip("igakit")
    from zkit.io.wfs_field import reconstruct_wfs

    p = tmp_path / "w_wfs.h5"
    _wfs_file(p, nfuncs)
    ev = _Evaluator(p, 23)
    with h5py.File(p) as f:
        res = ev(f["wavefunction"][1])
    ref = reconstruct_wfs(p, 1, axes=ev.axes)
    assert res["psi"].shape == tuple(len(a) for a in ev.axes)
    assert np.max(np.abs(res["psi"] - ref["psi"])) < 1e-12 * np.max(np.abs(ref["psi"]))


def test_paraview_reads_the_file(tmp_path):
    """Read a written file with ParaView's own VTK reader (skipped without ParaView)."""
    from zkit.viz.paraview import find_pvpython

    try:
        pv = find_pvpython()
    except FileNotFoundError:
        pytest.skip("ParaView not installed")
    x, y = np.linspace(-1, 1, 6), np.linspace(0, 2, 5)
    X, Y = np.meshgrid(x, y, indexing="ij")
    write_vti(tmp_path / "c.vti", [x, y], {"f": X * Y + 3}, {"TIME": 2.0})
    script = tmp_path / "read.py"
    script.write_text(
        "from paraview.simple import *\nfrom paraview import servermanager as sm\n"
        f"r = XMLImageDataReader(FileName=[{str(tmp_path / 'c.vti')!r}])\n"
        "d = sm.Fetch(r)\na = d.GetPointData().GetArray('f')\n"
        "i = d.FindPoint([0.2, 1.5, 0.0])\nprint('VAL', d.GetPoint(i), a.GetValue(i))\n"
        "print('TIME', d.GetFieldData().GetArray('TIME').GetValue(0))\n")
    out = subprocess.run([pv, str(script)], capture_output=True, text=True, timeout=300).stdout
    val = re.search(r"VAL \(([^)]+)\) (\S+)", out)
    px, py, _ = (float(v) for v in val.group(1).split(","))
    assert float(val.group(2)) == pytest.approx(px * py + 3, rel=1e-6)
    assert "TIME 2.0" in out

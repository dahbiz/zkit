from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

import zkit
from zkit.io.eigen import read_eigen
from zkit.io.wfs_field import _reshape_static_coefficients
from zkit.simulation import Run


def test_public_version_and_reference_energy() -> None:
    assert zkit.__version__ == "0.1.0b1"
    assert zkit.infinite_well_energy(1, 2.0) == pytest.approx(np.pi**2 / 8.0)


def test_partition_of_unity_inside_domain() -> None:
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    points = np.linspace(0.0, 1.0, endpoint=False)
    assert zkit.partition_of_unity(knots, 2, points) == pytest.approx(np.ones_like(points))


def test_run_parses_input_metadata() -> None:
    data_dir = Path(__file__).parent / "data"
    run = Run(data_dir, "test_deck.inp")

    assert run.meta.dimension == 2
    assert run.meta.time_step == pytest.approx(0.04)
    assert run.meta.final_time == pytest.approx(6849.0)
    assert run.meta.output_stride_wfs == 7000


def test_read_eigen_supports_tdsez_layout(tmp_path: Path) -> None:
    path = tmp_path / "EigenData_example.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("spectrum", data=np.array([[-0.5, 0.0], [0.25, 0.0]]))
        metadata = handle.create_group("run_metadata")
        metadata.attrs["Dimension"] = 1
        handle.create_dataset("psi_0", data=np.array([[1.0, 0.0], [0.0, 0.0]]))
        handle.create_dataset("psi_1", data=np.array([[0.0, 0.0], [1.0, 0.0]]))

    eigen = read_eigen(path)

    assert eigen.dimension == 1
    assert eigen.n_states == 2
    assert eigen.values == pytest.approx([-0.5, 0.25])
    assert eigen.vectors.shape == (2, 4)


def test_tdm_cli_normalizes_requested_axes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from zkit import cli

    eigen_path = tmp_path / "EigenData_example.h5"
    eigen_path.touch()
    received = {}

    def fake_plot_tdm(path, **kwargs):
        received.update(kwargs)
        return {"diagram": "diagram.png", "matrix": "matrix.png"}

    monkeypatch.setattr(cli, "plot_tdm", fake_plot_tdm)
    args = Namespace(
        eig_file=str(eigen_path),
        run_dir=".",
        axes="x, y",
        outdir=str(tmp_path),
        min_mu=1e-3,
        color_by="strength",
        dpi=150,
        prefix="tdm",
    )

    assert cli.cmd_tdm_plot(args) == 0
    assert received["axes"] == ["x", "y"]


def test_static_coefficients_accept_tdsez_tensor_and_flat_layouts() -> None:
    # TDSE-Z stores [y][x] (x fastest): nfuncs (x, y) = (3, 4) -> tensor (4, 3, 2)
    stored = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
    out = _reshape_static_coefficients(stored, [3, 4], "psi_0")
    assert out.shape == (3, 4, 2)
    assert out[2, 1, 0] == stored[1, 2, 0]  # c(x=2, y=1)
    flat = _reshape_static_coefficients(stored.reshape(12, 2), [3, 4], "psi_0")
    assert np.array_equal(flat, out)
    # 3D: (n_z, n_y, n_x)
    s3 = np.random.default_rng(0).random((5, 4, 3, 2))
    o3 = _reshape_static_coefficients(s3, [3, 4, 5], "psi_0")
    assert o3.shape == (3, 4, 5, 2) and o3[2, 1, 4, 1] == s3[4, 1, 2, 1]

    with pytest.raises(ValueError, match="incompatible"):
        _reshape_static_coefficients(np.zeros((3, 2)), [3, 4], "psi_0")


def test_bundled_igakit_evaluator_preserves_partition_of_unity() -> None:
    from igakit.igalib import bsp

    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    control_points = np.ones((3, 1))
    values = bsp.Evaluate1(2, knots, control_points, np.linspace(0.0, 1.0, 9))[:, 0]

    assert values == pytest.approx(np.ones(9))


def test_read_kh_and_multipole(tmp_path):
    """run_metadata of a Mode = kh, Multipole = 2 eigen file -> KHQuiver, MultipoleInfo."""
    import h5py

    from zkit.io import read_kh, read_multipole

    p = tmp_path / "kh_eigen.h5"
    t = np.linspace(0.1, 6.2, 8)
    w = np.full(8, 1 / 8)
    al = np.stack([np.sin(t), 0.01 * np.sin(2 * t), 0 * t], 1)
    with h5py.File(p, "w") as f:
        g = f.create_group("run_metadata")
        g["kh_times"], g["kh_weights"], g["kh_alpha"] = t, w, al
        g.attrs.update(
            {
                "KH_window_start": 0.0,
                "KH_window_end": 6.3,
                "KH_omega": 1.0,
                "Multipole": 2,
                "PropagationDirection_x": 0.0,
                "PropagationDirection_y": 1.0,
                "PropagationDirection_z": 0.0,
                "SpeedOfLight": 137.0,
                "MultipoleMagnetic": 1,
            }
        )
    kh = read_kh(p)
    assert kh.alpha.shape == (8, 3) and np.allclose(kh.alpha, al) and kh.window == (0.0, 6.3)
    x = np.linspace(-1, 1, 5)
    assert np.allclose(kh.average(lambda xx: xx, x), x + w @ al[:, 0])
    mp = read_multipole(p)
    assert (
        mp.order == 2
        and mp.direction == (0.0, 1.0, 0.0)
        and mp.speed_of_light == 137.0
        and mp.magnetic
    )
    q = tmp_path / "old.h5"
    with h5py.File(q, "w") as f:
        f.create_group("run_metadata")
    assert read_multipole(q).order is None


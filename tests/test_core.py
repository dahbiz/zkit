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
    tensor = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)

    assert _reshape_static_coefficients(tensor, [3, 4], "psi_0").shape == (3, 4, 2)
    assert _reshape_static_coefficients(tensor.reshape(12, 2), [3, 4], "psi_0").shape == (
        3,
        4,
        2,
    )

    with pytest.raises(ValueError, match="incompatible"):
        _reshape_static_coefficients(np.zeros((3, 2)), [3, 4], "psi_0")

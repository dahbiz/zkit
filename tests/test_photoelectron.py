"""t-SURFF / SCID-TDSE readers, photoelectron spectra and harmonic spectra."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from zkit.cli import main
from zkit.hhg import harmonic_spectrum
from zkit.io.scid import read_scid_detail, read_scid_pes
from zkit.io.surff import read_surff
from zkit.photoelectron import (
    angular_distribution,
    compare_spectra,
    energy_spectrum,
    region_yields,
    scid_energy_spectrum,
    to_spherical,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scid_benchmark"
K0, SIG = 0.6, 0.12


def shell(k):
    """Isotropic Gaussian shell dP/d^3k; total yield 4 pi int k^2 P dk."""
    return np.exp(-(((k - K0) / SIG) ** 2))


def shell_yield():
    k = np.linspace(0, 3, 200001)
    return float(np.sum(4 * np.pi * k**2 * shell(k)) * (k[1] - k[0]))


def write_surff(path, kx, ky, kz, density):
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    P = density(np.sqrt(KX**2 + KY**2 + KZ**2))
    mom = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], -1)
    amp = np.sqrt(P.ravel())
    with h5py.File(path, "w") as f:
        f["momenta"] = mom
        f["amplitude"] = np.stack([amp, 0 * amp], -1)
        f["pes"] = P.ravel()
        f["k_x"], f["k_y"], f["k_z"] = kx, ky, kz
        f["pes_grid"] = P
        f.attrs["radius"] = 11.0
        f.attrs["end_term"] = 1.0


@pytest.fixture
def plane_file(tmp_path):
    path = tmp_path / "surff_plane.h5"
    write_surff(path, np.linspace(0, 1.5, 151), np.zeros(1), np.linspace(-1.5, 1.5, 301), shell)
    return path


def test_read_surff_and_energy_spectrum_of_a_plane(plane_file) -> None:
    spec = read_surff(plane_file)
    assert spec.is_grid and spec.pes_grid.shape == (151, 1, 301)
    assert spec.attrs["radius"] == pytest.approx(11.0)
    assert spec.energy[-1] == pytest.approx((1.5**2 + 1.5**2) / 2)
    k = np.linspace(0.005, 1.4, 280)
    theta = (np.arange(60) + 0.5) * np.pi / 60
    P = to_spherical(spec, k, theta)
    assert P == pytest.approx(shell(k)[:, None] * np.ones(60), abs=2e-3)
    energy, dpde = energy_spectrum(k, theta, P)
    total = region_yields(energy, dpde, [(0, 2, "total")])["total"]
    assert total == pytest.approx(shell_yield(), rel=5e-3)


def test_to_spherical_on_a_cube_and_on_a_line(tmp_path) -> None:
    ax = np.linspace(-1.2, 1.2, 61)
    write_surff(tmp_path / "cube.h5", ax, ax, ax, shell)
    k = np.linspace(0.3, 0.9, 7)
    theta = np.array([0.3, 1.2, 2.5])
    P = to_spherical(read_surff(tmp_path / "cube.h5"), k, theta, phi=0.7)
    assert P == pytest.approx(shell(k)[:, None] * np.ones(3), abs=3e-2)  # trilinear, dk = 0.04
    with h5py.File(tmp_path / "line.h5", "w") as f:
        f["momenta"] = ax[:, None]
        f["amplitude"] = np.stack([ax, 0 * ax], -1)
        f["pes"] = ax**2
        f["k_x"], f["pes_grid"] = ax, ax**2
    line = read_surff(tmp_path / "line.h5")
    assert line.momenta.shape == (61, 1)
    P = to_spherical(line, np.array([0.5]), np.array([0.0, np.pi]))
    assert P.ravel() == pytest.approx([0.25, 0.25], rel=1e-2)


def write_scid_pes(path, k, theta_deg, density):
    rows = []
    for kk in k:
        for th in theta_deg:
            t = np.radians(th)
            a = np.sqrt(density(kk))
            rows.append([kk * np.sin(t), 0, kk * np.cos(t), kk, th, 0.0, a, 0.0, 0.0])
    np.savetxt(path, rows, header="Kx Ky Kz |K| Theta Phi Re[A] Im[A] Phi0")


def test_scid_pes_matches_an_identical_tdsez_spectrum(tmp_path, plane_file) -> None:
    k = np.linspace(0.01, 1.4, 140)
    th = (np.arange(37) + 0.5) * 180 / 37
    write_scid_pes(tmp_path / "pes.table", k, th, shell)
    scid = read_scid_pes(tmp_path / "pes.table")
    assert scid.amplitude.shape == (140, 37, 1)
    assert scid.theta == pytest.approx(np.radians(th))
    energy, ref = scid_energy_spectrum(scid)
    _, test = energy_spectrum(k, scid.theta, to_spherical(read_surff(plane_file), k, scid.theta))
    cmp = compare_spectra(energy, test, ref, [(0, 0.2, "low"), (0.2, 1.0, "peak")])
    assert cmp["ratio"]["low"] == pytest.approx(1.0, abs=1e-2)
    assert cmp["ratio"]["peak"] == pytest.approx(1.0, abs=1e-2)
    assert cmp["pointwise"][0] > 0.98 and cmp["pointwise"][1] < 1.02
    e, y = angular_distribution(energy, scid.density[:, :, 0], K0**2 / 2)
    assert e == pytest.approx(K0**2 / 2, abs=0.01) and y.max() == pytest.approx(1.0)


def test_read_scid_detail_uses_the_header_columns(tmp_path) -> None:
    titles = [
        "Timestep",
        "Time, au[t]",
        "Vector-potential magnitude",
        "Re[<Psi_L|Psi_R>]",
        "Re[<Psi_L|H_at+H_L+V_cap|Psi_R>], Hartree",
    ]
    titles += [f"Re[<Psi_L|e . {a}|Psi_R>], e-Bohr" for a in "xyz"]
    titles += [f"Re[(d^2/d t^2) <Psi_L|e . {a}|Psi_R>], e-Bohr/jiffy^2" for a in "xyz"]
    titles += [f"Re[(d/d t) <Psi_L|e . {a}|Psi_R>], e-Bohr/jiffy" for a in "xyz"]
    titles += [f"Electric field, lab {a}, atomic units" for a in "XYZ"]
    head = "\n".join(f"# {i + 1:2d} {i * 33 + 1},{i * 33 + 32}   {t}" for i, t in enumerate(titles))
    data = np.arange(3 * len(titles), dtype=float).reshape(3, -1)
    path = tmp_path / "detail.table"
    path.write_text(head + "\n" + "\n".join(" ".join(map(str, r)) for r in data) + "\n")
    d = read_scid_detail(path)
    assert d["time"] == pytest.approx(data[:, 1])
    assert d["dipole"][:, 2] == pytest.approx(data[:, 7])
    assert d["acceleration"][:, 0] == pytest.approx(data[:, 8])
    assert d["velocity"][:, 1] == pytest.approx(data[:, 12])
    assert d["field"][:, 2] == pytest.approx(data[:, 16])


def test_harmonic_spectrum_peaks_at_the_driven_order() -> None:
    t = np.arange(0, 400, 0.05)
    omega0 = 0.4
    d = np.sin(3 * omega0 * t)
    order, s = harmonic_spectrum(t, d, "dipole", omega0=omega0)
    assert order[np.argmax(s)] == pytest.approx(3.0, abs=0.02)
    _, s_acc = harmonic_spectrum(t, -((3 * omega0) ** 2) * d, "acceleration", omega0=omega0)
    i = np.argmax(s)
    assert s_acc[i] == pytest.approx(s[i], rel=5e-3)  # omega^4 off the exact bin
    with pytest.raises(ValueError):
        harmonic_spectrum(np.r_[t[:10], t[11:]], np.r_[d[:10], d[11:]])


@pytest.mark.skipif(not EXAMPLE.exists(), reason="benchmark data not present")
def test_scid_benchmark_reproduces_the_published_metrics(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from zkit.benchmark import compare_with_scid

    cmp = compare_with_scid(
        EXAMPLE / "tdsez",
        "quick.inp",
        EXAMPLE / "scid" / "detail.table.gz",
        EXAMPLE / "scid" / "pes_volkov.table.gz",
        omega=0.4,
        e0=0.05,
        regions=[(0.0, 1.2, "total"), (0.12, 0.47, "two-photon")],
    )
    m = cmp.metrics
    assert m["field"] < 1e-10
    assert m["dipole"] == pytest.approx(5.36e-3, rel=0.01)
    assert m["harmonics"] < 0.02
    assert m["yields"]["ratio"]["total"] == pytest.approx(1.032, abs=1e-3)
    assert m["yields"]["ratio"]["two-photon"] == pytest.approx(1.018, abs=1e-3)
    out = tmp_path / "cmp"
    rc = main(
        [
            "--run-dir",
            str(EXAMPLE / "tdsez"),
            "compare-scid",
            "quick.inp",
            "--scid-detail",
            str(EXAMPLE / "scid" / "detail.table.gz"),
            "--scid-pes",
            str(EXAMPLE / "scid" / "pes_volkov.table.gz"),
            "--omega",
            "0.4",
            "--e0",
            "0.05",
            "--outdir",
            str(out),
            "--dpi",
            "40",
        ]
    )
    assert rc == 0
    assert sorted(p.name for p in out.glob("*.png")) == [
        f"scid_{n}.png" for n in ("angular", "harmonics", "kmap", "pes", "time")
    ]


def test_cli_pes_writes_table_and_figure(tmp_path, plane_file) -> None:
    pytest.importorskip("matplotlib")
    assert main(["pes", str(plane_file), "--outdir", str(tmp_path), "--dpi", "40"]) == 0
    table = np.loadtxt(tmp_path / "pes_dPdE.txt")
    assert table.shape[1] == 2
    assert (tmp_path / "pes_dPdE.png").exists()

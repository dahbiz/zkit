"""Run the small 1D and 2D TDSEZ examples and plot their spectra."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from zkit.io.eigen import read_eigen
from zkit.io.evolution import read_evolution

CASES = {
    "1d": ("harmonic_oscillator_1d.inp", np.array([0.1, 0.3, 0.5]), 1),
    "2d": ("harmonic_oscillator_2d.inp", np.array([0.2, 0.4, 0.4, 0.6, 0.6, 0.6]), 2),
    "heterostructure": ("finite_quantum_well_heterostructure_1d.inp", None, 1),
    "rabi": ("ho1d_rabi.inp", None, 1),
    "3d": ("harmonic_oscillator_3d_small.inp", np.array([0.3006]), 3),
}


def configure_matplotlib() -> None:
    """Use a LaTeX-like serif style without requiring a TeX installation."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def plot_spectrum(values: np.ndarray, expected: np.ndarray, path: Path, title: str) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    indices = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(indices, values, label="TDSEZ", zorder=3)
    ax.scatter(indices, expected, marker="x", label="analytic", zorder=3)
    ax.set(xlabel=r"state index $n$", ylabel=r"energy $E_n$ (a.u.)", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heterostructure(values: np.ndarray, path: Path, profile_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    x = np.linspace(-15.0, 15.0, 1200)
    barrier = 0.6 * ((x < -4.0) | (x > 4.0))
    mass = 1.0 + 0.2 * ((x < -4.0) | (x > 4.0))
    np.savetxt(
        profile_path,
        np.column_stack((x, barrier, mass)),
        delimiter=",",
        header="x,potential,mass",
        comments="",
    )
    fig, (potential_ax, mass_ax) = plt.subplots(
        2, 1, figsize=(6, 5), sharex=True, height_ratios=(3, 1)
    )
    potential_ax.plot(x, barrier, color="tab:blue", label="conduction-band profile")
    for index, energy in enumerate(values):
        potential_ax.axhline(energy, color="tab:red", alpha=0.7, linewidth=0.9)
        potential_ax.text(14.6, energy, f"E{index}", ha="right", va="bottom", fontsize=8)
    potential_ax.set_ylabel("energy (a.u.)")
    potential_ax.set_title("Finite quantum well heterostructure")
    potential_ax.set_ylim(-0.03, 0.72)
    potential_ax.grid(alpha=0.25)
    potential_ax.legend(loc="upper right")
    mass_ax.plot(x, mass, color="tab:orange")
    mass_ax.set(xlabel="position (a.u.)", ylabel="m*(x)")
    mass_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rabi(evolution_path: Path, output_path: Path, csv_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    evolution = read_evolution(evolution_path)
    populations = evolution.populations[:, 1:]
    field = evolution.dipoles[:, 1]
    dipole = evolution.dipoles[:, 2]
    np.savetxt(
        csv_path,
        np.column_stack((evolution.time, populations, field, dipole)),
        delimiter=",",
        header="time,p0,p1,p2,field_x,dipole_x",
        comments="",
    )
    plot_observables(evolution, case_dir=output_path.parent)
    fig, (population_ax, response_ax) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for state, values in enumerate(populations.T):
        population_ax.plot(evolution.time, values, label=f"P{state}")
    # Two-level rotating-wave estimate for comparison with the full propagation.
    omega = 0.2
    field_amplitude = 0.05
    dipole_01 = np.sqrt(1.0 / (2.0 * omega))
    rabi_reference = np.sin(0.25 * field_amplitude * dipole_01 * evolution.time) ** 2
    population_ax.plot(
        evolution.time,
        rabi_reference,
        "k--",
        linewidth=1.0,
        label=r"two-level RWA $\sin^2(\Omega_R t/2)$",
    )
    population_ax.set_ylabel("bound-state population")
    population_ax.set_ylim(-0.02, 1.05)
    population_ax.grid(alpha=0.25)
    population_ax.legend(ncol=3)
    response_ax.plot(evolution.time, field, label=r"field $E_x(t)$", color="tab:orange")
    response_ax.plot(evolution.time, dipole, label=r"dipole $\langle x\rangle$", color="tab:blue")
    response_ax.set(xlabel=r"time $t$ (a.u.)", ylabel="response (a.u.)")
    response_ax.grid(alpha=0.25)
    response_ax.legend()
    fig.suptitle("Driven 1D oscillator: Rabi-style population transfer")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_observables(evolution, case_dir: Path) -> None:
    """Plot energy conservation, current decomposition, and spectra."""
    configure_matplotlib()
    import matplotlib.pyplot as plt

    time = evolution.time
    energy = evolution.energies
    current = evolution.currents
    autocorrelation = evolution.autocorrelation[:, 1] + 1j * evolution.autocorrelation[:, 2]
    dt = float(np.mean(np.diff(time)))
    frequency = 2.0 * np.pi * np.fft.rfftfreq(time.size, dt)
    dipole = evolution.dipoles[:, 2] - np.mean(evolution.dipoles[:, 2])
    dipole_spectrum = np.abs(np.fft.rfft(dipole)) ** 2
    autocorrelation_spectrum = np.abs(np.fft.fft(autocorrelation)[: frequency.size])

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=False)
    axes[0].plot(time, energy[:, 1], label=r"kinetic $\langle K\rangle$")
    axes[0].plot(time, energy[:, 2], label=r"potential $\langle V\rangle$")
    axes[0].plot(time, energy[:, 3], label=r"interaction $\langle H_{int}\rangle$")
    axes[0].plot(time, energy[:, 4], label=r"total $\langle H\rangle$", linewidth=2)
    axes[0].set_ylabel(r"energy (a.u.)")
    axes[0].set_title("Energy decomposition")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].plot(time, current[:, 3], label=r"total $J_x$")
    axes[1].plot(time, current[:, 6], label=r"intra-band $J_x$")
    axes[1].plot(time, current[:, 9], label=r"inter-band $J_x$")
    axes[1].set(
        xlabel=r"time $t$ (a.u.)", ylabel=r"current $J_x$ (a.u.)", title="Current decomposition"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[2].plot(frequency, dipole_spectrum / max(dipole_spectrum.max(), 1e-30), label="dipole FFT")
    axes[2].plot(
        frequency,
        autocorrelation_spectrum / max(autocorrelation_spectrum.max(), 1e-30),
        label="autocorrelation spectrum",
    )
    axes[2].set(
        xlabel=r"angular frequency $\omega$ (a.u.)",
        ylabel="normalized amplitude",
        title="Response spectra",
    )
    axes[2].set_xlim(0, 1.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(case_dir / "energy-current-spectra.png", dpi=150)
    plt.close(fig)


def plot_3d_eigenstate(eigen, output_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    field = eigen.reconstruct(istate=0, npoints=32)
    middle = field["abs2"].shape[2] // 2
    extent = [field["axes"][0][0], field["axes"][0][-1], field["axes"][1][0], field["axes"][1][-1]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(field["abs2"][:, :, middle].T, origin="lower", extent=extent, cmap="magma")
    ax.set(xlabel=r"$x$ (a.u.)", ylabel=r"$y$ (a.u.)", title=r"$|\psi_0(x,y,z=0)|^2$")
    fig.colorbar(image, ax=ax, label=r"$|\psi_0|^2$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_1d_eigenstate(eigen, output_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    field = eigen.reconstruct(istate=0, npoints=500)
    x = field["axes"][0]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    axes[0].plot(x, field["Re"], label=r"$\operatorname{Re}\psi_0$")
    axes[0].plot(x, field["Im"], label=r"$\operatorname{Im}\psi_0$")
    axes[0].plot(x, field["abs2"], label=r"$|\psi_0|^2$", linewidth=2)
    axes[0].set_ylabel("amplitude / density")
    axes[0].set_title(r"1D ground state: $\psi_0(x)$")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    phase = np.unwrap(np.angle(field["psi"]))
    axes[1].plot(x, phase, color="tab:purple")
    axes[1].set(xlabel=r"position $x$ (a.u.)", ylabel=r"phase $\arg\psi_0$")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_2d_eigenstate(eigen, output_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    field = eigen.reconstruct(istate=0, npoints=100)
    x, y = field["axes"]
    extent = [x[0], x[-1], y[0], y[-1]]
    density = field["abs2"].T
    phase = np.angle(field["psi"]).T
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    image = axes[0].imshow(density, origin="lower", extent=extent, cmap="magma")
    axes[0].set_title(r"density $|\psi_0(x,y)|^2$")
    fig.colorbar(image, ax=axes[0], shrink=0.85)
    image = axes[1].imshow(phase, origin="lower", extent=extent, cmap="twilight")
    axes[1].set_title(r"phase $\arg\psi_0(x,y)$")
    fig.colorbar(image, ax=axes[1], shrink=0.85)
    levels = np.linspace(float(density.min()), float(density.max()), 12)
    axes[2].contourf(density, levels=levels, extent=extent, cmap="viridis")
    axes[2].contour(density, levels=levels, extent=extent, colors="white", linewidths=0.35)
    axes[2].set_title(r"density contours")
    for axis in axes:
        axis.set(xlabel=r"$x$ (a.u.)", ylabel=r"$y$ (a.u.)")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_case(tdsez: Path, case: str, output_dir: Path) -> None:
    deck_name, expected, dimension = CASES[case]
    source = Path(__file__).parent / "inputs" / deck_name
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    deck = case_dir / deck_name
    shutil.copy2(source, deck)

    started = time.perf_counter()
    result = subprocess.run(
        [str(tdsez), "-inp", deck.name],
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(
            f"TDSEZ failed for {case} (exit {result.returncode})\n"
            f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )

    eigen_path = case_dir / "static" / f"EigenData_{deck.name}.h5"
    eigen = read_eigen(eigen_path)
    if eigen.dimension != dimension or (
        expected is not None and not np.allclose(eigen.values, expected, atol=2e-5)
    ):
        raise RuntimeError(
            f"Unexpected {case} spectrum: dimension={eigen.dimension}, values={eigen.values}"
        )

    np.savetxt(
        case_dir / "spectrum.csv",
        np.column_stack((np.arange(eigen.n_states), eigen.values)),
        delimiter=",",
        header="state,energy",
        comments="",
    )
    if case == "1d":
        plot_1d_eigenstate(eigen, case_dir / "ground-state-components.png")
    elif case == "2d":
        plot_2d_eigenstate(eigen, case_dir / "ground-state-density-phase-contours.png")
    elif case == "3d":
        plot_3d_eigenstate(eigen, case_dir / "ground-state-central-slice.png")
    elif case == "heterostructure":
        plot_heterostructure(
            eigen.values,
            case_dir / "potential_and_levels.png",
            case_dir / "profile.csv",
        )
    elif case == "rabi":
        plot_rabi(
            case_dir / "td" / f"TimeEvolutionData_{deck.name}.h5",
            case_dir / "rabi-populations-response.png",
            case_dir / "rabi-observables.csv",
        )
    else:
        plot_spectrum(
            eigen.values,
            expected,
            case_dir / "spectrum.png",
            f"TDSEZ {case} harmonic oscillator",
        )
    error = (
        f", max error {np.max(np.abs(eigen.values - expected)):.2e}" if expected is not None else ""
    )
    print(f"{case}: {eigen.n_states} states{error}, {elapsed:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdsez", type=Path, required=True, help="path to the TDSEZ executable")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument(
        "--case", choices=["1d", "2d", "3d", "heterostructure", "rabi", "all"], default="all"
    )
    args = parser.parse_args()
    if not args.tdsez.is_file():
        parser.error(f"TDSEZ executable not found: {args.tdsez}")
    cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    for case in cases:
        run_case(args.tdsez.resolve(), case, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

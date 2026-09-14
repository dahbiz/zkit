"""Run the small 1D and 2D TDSEZ examples and plot their spectra."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from zkit.io.eigen import read_eigen

CASES = {
    "1d": ("harmonic_oscillator_1d.inp", np.array([0.1, 0.3, 0.5]), 1),
    "2d": ("harmonic_oscillator_2d.inp", np.array([0.2, 0.4, 0.4, 0.6, 0.6, 0.6]), 2),
}


def plot_spectrum(values: np.ndarray, expected: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(indices, values, label="TDSEZ", zorder=3)
    ax.scatter(indices, expected, marker="x", label="analytic", zorder=3)
    ax.set(xlabel="state index", ylabel="energy (a.u.)", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
    if eigen.dimension != dimension or not np.allclose(eigen.values, expected, atol=2e-5):
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
    plot_spectrum(
        eigen.values,
        expected,
        case_dir / "spectrum.png",
        f"TDSEZ {case} harmonic oscillator",
    )
    print(
        f"{case}: {eigen.n_states} states, max error {np.max(np.abs(eigen.values - expected)):.2e}, {elapsed:.2f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdsez", type=Path, required=True, help="path to the TDSEZ executable")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--case", choices=["1d", "2d", "all"], default="all")
    args = parser.parse_args()
    if not args.tdsez.is_file():
        parser.error(f"TDSEZ executable not found: {args.tdsez}")
    cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    for case in cases:
        run_case(args.tdsez.resolve(), case, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CLI for reading TDSEZ simulation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zkit import Run
from zkit.io.eigen import read_eigen
from zkit.io.wfs_field import compute_wfs_norm
from zkit.viz.tdm import plot_tdm
from zkit.viz.wavefunction import plot_wavefunction


def _summarize_evolution(ev) -> str:
    return (
        f"dimension={ev.dimension} "
        f"steps={ev.n_steps} "
        f"dipoles={ev.dipoles.shape} "
        f"populations={ev.populations.shape} "
        f"energies={ev.energies.shape}"
    )


def cmd_summary(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    base = args.basename
    sim = Run(run_dir, base)
    meta = sim.meta
    ev = sim.evolution

    eigen_path = run_dir / "static" / f"EigenData_{base}.h5"
    eigen_text = "missing"
    if eigen_path.exists():
        try:
            eigen = read_eigen(eigen_path)
            eigen_text = f"states={int(eigen.values.shape[0])}"
        except Exception as exc:  # pragma: no cover - CLI path only
            eigen_text = f"error={exc}"

    print(f"input={base}")
    print(f"dimension={meta.dimension}")
    print(f"n_steps={ev.n_steps}")
    print(f"time_step={meta.time_step}")
    print(f"final_time={meta.final_time}")
    print(f"evolution: {_summarize_evolution(ev)}")
    print(f"eigen: {eigen_text}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    td_dir = run_dir / "td"
    if not td_dir.is_dir():
        print(f"No td directory found in {run_dir}", file=sys.stderr)
        return 1

    files = sorted(td_dir.glob("TimeEvolutionData_*.h5"))
    if not files:
        print(f"No TimeEvolutionData_*.h5 files found in {td_dir}", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        base = path.name[len("TimeEvolutionData_") : -len(".h5")]
        try:
            sim = Run(run_dir, base)
            ev = sim.evolution
            rows.append((base, sim.meta.dimension, ev.n_steps, ev.time[-1] if ev.n_steps else 0.0))
        except Exception as exc:  # pragma: no cover - CLI path only
            rows.append((base, "error", 0, str(exc)))

    print("basename,dimension,steps,final_time")
    for base, dim, steps, final_time in rows:
        if isinstance(final_time, float):
            print(f"{base},{dim},{steps},{final_time}")
        else:
            print(f"{base},{dim},0,{final_time}")

    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    out = plot_wavefunction(
        args.wfs_file,
        step=args.step,
        outdir=args.outdir,
        npoints=args.npoints,
        dpi=args.dpi,
        vtk=args.vtk,
    )
    print(f"wrote {out}")
    if args.norm:
        nrm = compute_wfs_norm(args.wfs_file, step=args.step, nq=5)
        print(f"exact norm <psi|psi> = {nrm:.12f}")
    return 0


def cmd_tdm_plot(args: argparse.Namespace) -> int:
    eig = Path(args.eig_file)
    # Accept either the EigenData_<base>.h5 directly, or a run-dir + basename.
    if eig.is_file():
        eig_path = eig
    else:
        run_dir = Path(args.run_dir)
        cand = run_dir / "static" / f"EigenData_{eig.name}.h5"
        if not cand.exists():
            cand = run_dir / "static" / f"EigenData_{eig}.h5"
        if not cand.exists():
            print(f"EigenData HDF5 not found for {args.eig_file}", file=sys.stderr)
            return 1
        eig_path = cand
    axes = None
    if args.axes:
        axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    out = plot_tdm(
        eig_path,
        outdir=Path(args.outdir),
        axes=axes,
        min_mu=args.min_mu,
        color_by=args.color_by,
        dpi=args.dpi,
        prefix=args.prefix,
    )
    print(f"wrote diagram -> {out['diagram']}")
    print(f"wrote matrix  -> {out['matrix']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read TDSEZ simulation outputs.")
    parser.add_argument("--run-dir", default=".", help="Run directory containing td/ and static/")

    subparsers = parser.add_subparsers(dest="command")

    p_summary = subparsers.add_parser("summary", help="Summarize one simulation")
    p_summary.add_argument("basename", help="Input basename such as h2p.inp")

    subparsers.add_parser("batch", help="Summarize all TimeEvolutionData files in td/")

    p_plot = subparsers.add_parser("plot-wfs", help="Plot a wavefunction snapshot to PNG")
    p_plot.add_argument("wfs_file", help="Path to td/wfs_<input>.h5")
    p_plot.add_argument("--step", type=int, default=0, help="Snapshot index")
    p_plot.add_argument("--outdir", default=".", help="Output directory")
    p_plot.add_argument("--npoints", type=int, default=200, help="Grid points per axis")
    p_plot.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    p_plot.add_argument(
        "--vtk", default=None, help="Also export to VTK for ParaView (e.g. out.vts / out.vtu)"
    )
    p_plot.add_argument(
        "--norm",
        action="store_true",
        help="Print the exact <psi|psi> norm via Gauss quadrature "
        "over the knot spans (igakit basis engine)",
    )

    p_tdm = subparsers.add_parser(
        "tdm-plot", help="Plot transition-dipole diagram + |mu_ij| heatmap from EigenData HDF5"
    )
    p_tdm.add_argument(
        "eig_file", help="EigenData_<base>.h5 path, or just the basename when used with --run-dir"
    )
    p_tdm.add_argument(
        "--run-dir",
        default=".",
        help="Run directory holding static/ (used if eig_file is a basename)",
    )
    p_tdm.add_argument(
        "--axes",
        default=None,
        help="Comma-separated dipole axes to include, e.g. x,y,z (default: all present)",
    )
    p_tdm.add_argument(
        "--min-mu", type=float, default=1e-3, help="Minimum |mu_ij| to draw a coupling arrow"
    )
    p_tdm.add_argument(
        "--color-by",
        default="strength",
        choices=["strength", "f"],
        help="Arrow colour: 'strength' (|mu|) or 'f' (oscillator strength)",
    )
    p_tdm.add_argument("--outdir", default=".", help="Output directory")
    p_tdm.add_argument("--prefix", default="tdm", help="Output filename prefix")
    p_tdm.add_argument("--dpi", type=int, default=150, help="Figure DPI")

    args = parser.parse_args(argv)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "batch":
        return cmd_batch(args)
    if args.command == "plot-wfs":
        return cmd_plot(args)
    if args.command == "tdm-plot":
        return cmd_tdm_plot(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

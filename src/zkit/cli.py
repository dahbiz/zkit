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


def cmd_validate(args: argparse.Namespace) -> int:
    from zkit.validation import default_binary, validate

    binary = args.binary or default_binary()
    if not binary:
        print("no TDSE-Z binary: pass --binary or set TDSEZ_BINARY")
        return 2
    report = validate(binary, args.reference, args.np, args.workdir, quick=args.quick)
    print(report.text())
    return 0 if report.ok else 1


def _regions(text: str | None) -> list | None:
    if not text:
        return None
    out = [(0.0, float("inf"), "total")]
    for part in text.split(","):
        lo, hi = (float(v) for v in part.split(":"))
        out.append((lo, hi, f"E {lo:g}-{hi:g}"))
    return out


def cmd_pes(args: argparse.Namespace) -> int:
    import numpy as np

    from zkit.io.scid import read_scid_pes
    from zkit.io.surff import read_surff
    from zkit.photoelectron import energy_spectrum, scid_energy_spectrum, to_spherical

    spec = read_surff(args.surff_file)
    curves = {}
    if args.scid:
        scid = read_scid_pes(args.scid)
        k, theta = scid.k, scid.theta
        energy, curves["SCID-TDSE"] = scid_energy_spectrum(scid)
    else:
        kmax = float(np.max(np.abs(np.concatenate([np.ravel(a) for a in spec.axes or ()]))))
        k = np.linspace(kmax / args.nk, kmax, args.nk)
        theta = (np.arange(args.ntheta) + 0.5) * np.pi / args.ntheta
    energy, dpde = energy_spectrum(k, theta, to_spherical(spec, k, theta))
    curves = {"TDSE-Z": dpde, **curves}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table = outdir / f"{args.prefix}_dPdE.txt"
    header = "E (a.u.)  " + "  ".join(f"dP/dE {lab}" for lab in curves)
    np.savetxt(table, np.column_stack([energy, *curves.values()]), header=header)
    print(f"yield {args.prefix}: " + ", ".join(
        f"{lab} {float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(energy))):.4e}"
        for lab, y in curves.items()
    ))  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    from zkit.viz.photoelectron import plot_pes

    fig = plot_pes(energy, curves, reference=list(curves)[-1])
    fig.savefig(outdir / f"{args.prefix}_dPdE.png", dpi=args.dpi)
    print(f"wrote {table} and {outdir / (args.prefix + '_dPdE.png')}")
    return 0


def cmd_compare_scid(args: argparse.Namespace) -> int:
    from zkit.benchmark import compare_with_scid, plot_comparison

    cmp = compare_with_scid(
        args.run_dir,
        args.basename,
        args.scid_detail,
        args.scid_pes,
        omega=args.omega,
        e0=args.e0,
        ip=args.ip,
        regions=_regions(args.regions),
    )
    text = cmp.report()
    print(text)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.prefix}_metrics.txt").write_text(text + "\n")
    surff = Path(args.run_dir) / "td" / f"surff_{Path(args.basename).name}.h5"
    for p in plot_comparison(cmp, outdir, surff, args.scid_pes, args.prefix, args.dpi):
        print(f"wrote {p}")
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

    p_val = subparsers.add_parser(
        "validate", help="Run TDSE-Z on analytic cases and check every output column"
    )
    p_val.add_argument("--binary", default=None, help="tdsez.x (default: TDSEZ_BINARY or PATH)")
    p_val.add_argument("--reference", default=None, help="Second binary: compare every column")
    p_val.add_argument("--np", type=int, default=4, help="MPI ranks per run")
    p_val.add_argument("--workdir", default="zkit-validation", help="Directory for the runs")
    p_val.add_argument("--quick", action="store_true", help="2D cases only (about a minute)")

    p_pes = subparsers.add_parser(
        "pes", help="dP/dE from a t-SURFF file (optionally against a SCID-TDSE table)"
    )
    p_pes.add_argument("surff_file", help="Path to td/surff_<input>.h5 (grid mode)")
    p_pes.add_argument("--scid", default=None, help="SCID-TDSE sts_volkov/coulomb table")
    p_pes.add_argument("--nk", type=int, default=300, help="|k| points (without --scid)")
    p_pes.add_argument("--ntheta", type=int, default=90, help="theta points (without --scid)")
    p_pes.add_argument("--outdir", default=".", help="Output directory")
    p_pes.add_argument("--prefix", default="pes", help="Output filename prefix")
    p_pes.add_argument("--dpi", type=int, default=150, help="Figure DPI")

    p_cmp = subparsers.add_parser(
        "compare-scid", help="Compare a TDSE-Z run with SCID-TDSE: metrics and figures"
    )
    p_cmp.add_argument("basename", help="TDSE-Z input basename, e.g. quick.inp")
    p_cmp.add_argument("--scid-detail", required=True, help="SCID-TDSE detail_output table")
    p_cmp.add_argument("--scid-pes", required=True, help="SCID-TDSE sts_volkov_table")
    p_cmp.add_argument("--omega", type=float, required=True, help="Laser frequency (a.u.)")
    p_cmp.add_argument("--e0", type=float, required=True, help="Peak field (a.u.)")
    p_cmp.add_argument("--ip", type=float, default=0.5, help="Ionisation potential (a.u.)")
    p_cmp.add_argument(
        "--regions", default=None, help="Energy windows for yields, e.g. 0:0.12,0.12:0.47"
    )
    p_cmp.add_argument("--outdir", default=".", help="Output directory")
    p_cmp.add_argument("--prefix", default="scid", help="Output filename prefix")
    p_cmp.add_argument("--dpi", type=int, default=180, help="Figure DPI")

    args = parser.parse_args(argv)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "batch":
        return cmd_batch(args)
    if args.command == "plot-wfs":
        return cmd_plot(args)
    if args.command == "tdm-plot":
        return cmd_tdm_plot(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "pes":
        return cmd_pes(args)
    if args.command == "compare-scid":
        return cmd_compare_scid(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

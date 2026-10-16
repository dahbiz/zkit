"""Reproduce the TDSE-Z vs SCID-TDSE benchmark figures and metrics with zkit.

    python reproduce.py            # writes figures/ and metrics.txt here

Same as:

    zkit --run-dir tdsez compare-scid quick.inp \
        --scid-detail scid/detail.table.gz --scid-pes scid/pes_volkov.table.gz \
        --omega 0.4 --e0 0.05 --regions 0:0.12,0.12:0.47,0.47:0.9,0.9:1.2 \
        --outdir figures
"""

from pathlib import Path

from zkit.benchmark import compare_with_scid, plot_comparison

HERE = Path(__file__).resolve().parent
REGIONS = [
    (0.0, 1.2, "total"),
    (0.0, 0.12, "E < 0.12 (below CAP k_min)"),
    (0.12, 0.47, "two-photon order"),
    (0.47, 0.9, "three-photon region"),
    (0.9, 1.2, "E > 0.9"),
]

cmp = compare_with_scid(
    HERE / "tdsez",
    "quick.inp",
    HERE / "scid" / "detail.table.gz",
    HERE / "scid" / "pes_volkov.table.gz",
    omega=0.4,
    e0=0.05,
    ip=0.5,
    regions=REGIONS,
)
text = cmp.report()
(HERE / "metrics.txt").write_text(text + "\n")
print(text)
surff = HERE / "tdsez" / "td" / "surff_quick.inp.h5"
for p in plot_comparison(cmp, HERE / "figures", surff, HERE / "scid" / "pes_volkov.table.gz"):
    print("wrote", p.relative_to(HERE))

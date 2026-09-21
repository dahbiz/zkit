# zkit — TDSEZ post-processing toolkit

`zkit` reads the HDF5 outputs written by the **TDSEZ** solver
(Time-Dependent Schrödinger Equation solver, B-spline / IGA) and provides
reconstruction, analytic-reference, and visualization helpers.

It is the analysis half of TDSEZ: the C++ binary produces `EigenData_*.h5`
/ `wfs_*.h5` / `TimeEvolutionData_*.h5`, and `zkit` turns those into spectra,
wavefunctions, transition-dipole diagrams, and Wigner transforms.

> **Status: BETA (0.1.0b1).** APIs may change between beta releases. File
> formats written by the TDSEZ binary are the stable contract.

---

## Install

### From source (recommended for now)

```bash
cd zkit
pip install -e .        # PyPI name is "zkit-lib"; import name stays "zkit"
```

This installs the `zkit` import package and the `zkit` command-line tool.

### Optional: Wigner-transform subpackage

The `zkit.mwigner` module (Wigner quasi-probability transforms) needs PyTorch.
It is **not** a core dependency — install it only if you need it:

```bash
pip install -e ".[mwigner]"
```

### Without installing (PYTHONPATH fallback)

If you just want to run the tools or test suite without installing, point
`PYTHONPATH` at the `src/` directory:

```bash
export PYTHONPATH=/path/to/zkit/src:$PYTHONPATH
python -m zkit --help
```

The import package lives at `src/zkit/`.

---

## Command-line tool

```bash
zkit summary  <basename>            # summarize one simulation (needs --run-dir)
zkit batch                         # summarize every TimeEvolutionData_*.h5 in td/
zkit plot-wfs <td/wfs_base.h5>     # render a wavefunction snapshot to PNG (+--vtk)
zkit tdm-plot <EigenData_base.h5>  # transition-dipole diagram + |mu_ij| heatmap
```

Run `zkit --help` (or `python -m zkit --help`) for all options.

## Quick TDSEZ examples

After building the `tdsez` executable, run the small static 1D and 2D harmonic
oscillator examples (each usually completes in a few seconds):

```bash
pip install -e ".[viz]"
python examples/run_quick_examples.py --tdsez /path/to/tdsez
```

Use `--case 1d` or `--case 2d` to run one case. Results are written to
`examples/output/` and include an HDF5 eigenvalue file, CSV spectrum, and PNG
plot. The input decks are in `examples/inputs/`.

---

## Python API

```python
import zkit

# High-level: load a whole run (eigen + time-evolution) from a directory
sim = zkit.load("run_dir", "h2p.inp")  # or zkit.Run(run_dir, base)
E = sim.eigen.values  # converged eigenvalues (a.u.)
ev = sim.evolution  # dipoles / populations / energies

# Low-level: open a single eigen HDF5
d = zkit.open_eig("static/EigenData_h2p.inp.h5")
print(d["spectrum"], d["meta"], d["knots_x"])

# Knot-vector sanity: reconstructed B-splines must sum to 1 (partition of unity)
kv = zkit.reconstruct_knots(d["knots_x"], p=int(d["meta"]["SplineDegree"]))
pou = zkit.partition_of_unity(kv, p, xs)
```

Key helpers:
- `zkit.open_eig`, `zkit.reconstruct_knots`, `zkit.partition_of_unity`
- `zkit.io` — readers for eigen / evolution / timeseries / wfs / tdm
- `zkit.viz` — `plot_wavefunction`, `plot_tdm`, `plot_transition_diagram`
- `zkit.mwigner` — Wigner-transform analysis (requires the `mwigner` extra)

---

## Units

TDSEZ code units: `hbar = 1`, `m = 1` ⇒ `hbar²/2m = 1/2`.
Energies are reported in atomic units (a.u.). See `zkit.__init__` for the
constants and the harmonic-oscillator / infinite-well reference formulas used
by the validation suite.

---

## License

BSD 2-Clause — see [LICENSE](LICENSE).

Author: Zakaria Dahbi (King's College London, Attosecond Quantum Physics Lab)
— zdahbi@outlook.es

## Acknowledgement

`zkit` ships a lightweight, pure-Python `igakit.igalib.bsp` compatibility
evaluator for IGA B-spline wavefunction reconstruction. It preserves the
upstream evaluator interface while avoiding a separate compiled dependency.
The bundled upstream BSD license is in `src/igakit/LICENSE.rst`.

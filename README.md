<p align="center">
  <img src="https://raw.githubusercontent.com/dahbiz/zkit/main/docs/_static/logo.png" alt="zkit" width="180">
</p>

<p align="center">
  <strong>Analysis toolkit for TDSE-Z simulations</strong><br>
  Read, reconstruct, validate and visualise the output of the TDSE-Z solver
</p>

<p align="center">
  <a href="https://github.com/dahbiz/zkit/actions/workflows/ci.yml"><img src="https://github.com/dahbiz/zkit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/dahbiz/zkit/actions/workflows/docs.yml"><img src="https://github.com/dahbiz/zkit/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a>
  <a href="https://dahbiz.github.io/zkit/"><img src="https://img.shields.io/badge/docs-dahbiz.github.io%2Fzkit-3b6ea5" alt="Docs"></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3b6ea5" alt="Python 3.10-3.13">
  <img src="https://img.shields.io/badge/license-BSD--2--Clause-3b6ea5" alt="License">
</p>

<p align="center"><sub><em>Z is not the author’s initial. Z is not the third axis.<br>
Z is ⵣ (yaz), the Amazigh sign of the free man and the free people.</em></sub></p>

---

`zkit` is the Python companion of [TDSE-Z](https://github.com/dahbiz/tdsez), a
B-spline (isogeometric) solver for the time-dependent Schrödinger equation in
1D, 2D and 3D. TDSE-Z writes HDF5 files; `zkit` turns them into spectra,
observables, wavefunctions, transition-dipole diagrams and Wigner
distributions, and validates the solver against exact solutions.

**Documentation:** https://dahbiz.github.io/zkit/

## Installation

`zkit` needs Python 3.10 or newer, `numpy` and `h5py`.

```bash
pip install zkit-lib
# the latest main branch
pip install "git+https://github.com/dahbiz/zkit.git"
# or, for development
git clone https://github.com/dahbiz/zkit.git
cd zkit
pip install -e ".[test]"
```

The distribution name is `zkit-lib`; the import name and the command are
`zkit`. Optional extras:

| Extra | Adds |
|---|---|
| `viz` | plotting (matplotlib) |
| `vtk` | VTK export of wavefunctions (pyvista) |
| `physics` | symbolic field expressions (sympy) |
| `validation` | the `zkit validate` suite (scipy, sympy) |
| `mwigner` | Wigner-function analysis (PyTorch, matplotlib) |
| `docs` | building the documentation (Sphinx) |

Install extras with, for example, `pip install -e ".[viz,validation]"`.

## Command-line tool

```bash
zkit --run-dir RUN summary <input>     # summary of one run (spectrum, time series)
zkit --run-dir RUN batch               # summary of every run in RUN/td/
zkit plot-wfs RUN/td/<name>_wfs.h5    # wavefunction snapshot to PNG (--vtk for VTK)
zkit tdm-plot RUN/static/<name>_eigen.h5   # transition-dipole diagram and |mu_ij| map
zkit validate --np 8                   # validate a TDSE-Z build against exact solutions
zkit pes RUN/td/<name>_surff.h5       # photoelectron spectrum dP/dE from t-SURFF
zkit --run-dir RUN compare-scid <input> --scid-detail D --scid-pes P --omega W --e0 E
                                       # one-to-one benchmark against SCID-TDSE
```

`examples/scid_benchmark/` holds a complete TDSE-Z vs SCID-TDSE benchmark
(inputs, outputs, figures and metrics) reproduced with `compare-scid`.

`<input>` is the name of the TDSE-Z input file, e.g. `h2p.inp`. Run
`zkit <command> --help` for the options of each command.

## Python API

```python
import numpy as np
import zkit

run = zkit.load("RUN", "h2p.inp")  # a TDSE-Z run directory and its input file
E = run.spectrum.values  # bound-state energies (a.u.)
t, D = run.time, run.dipoles  # time series of td/<name>_observables.h5
ph = run.physics  # carrier sign, fields E(t) from the input deck

# B-spline basis of a run: the functions sum to one inside the domain
d = zkit.open_eig("RUN/static/h2p_eigen.h5")
p = int(d["meta"]["SplineDegree"])
kv = zkit.reconstruct_knots(d["knots_x"], p)
xs = np.linspace(kv[0], kv[-1], 201)[1:-1]
assert np.allclose(zkit.partition_of_unity(kv, p, xs), 1.0)
```

Main modules:

- `zkit.io`: readers for the eigen, time-evolution, wavefunction and
  transition-dipole files;
- `zkit.simulation`: the `Run` object for a whole run directory;
- `zkit.physics`: carrier sign, field expressions and their integrals and
  derivatives, read exactly as TDSE-Z reads the input;
- `zkit.validation`: analytic references and the validation suite;
- `zkit.viz`: wavefunction and transition-dipole plots;
- `zkit.mwigner`: Wigner distributions, phase-space currents and covariances.

## Validating a TDSE-Z build

```bash
export TDSEZ_BINARY=/path/to/tdsez.x
zkit validate --quick                        # 2D cases, about 20 s
zkit validate --np 8                         # all cases
zkit validate --np 8 --reference old/tdsez.x # also compare two builds column by column
```

The suite runs free wave packets in a field, a driven oscillator (exact coherent
state) and position-dependent-mass problems, and checks every column of every
output table against exact solutions or exact identities; see the
[validation page](https://dahbiz.github.io/zkit/validation.html).

## Examples

With a TDSE-Z build, the static 1D and 2D harmonic-oscillator examples run in a
few seconds:

```bash
pip install -e ".[viz]"
python examples/run_quick_examples.py --tdsez /path/to/tdsez.x
```

Results (HDF5 spectrum, CSV, PNG plot) are written to `examples/output/`; the
input decks are in `examples/inputs/`. Larger campaigns are described in the
[documentation](https://dahbiz.github.io/zkit/examples/index.html).

## Units

Atomic units throughout (ħ = m_e = e = 1). The mass, charge and fields are
those of the TDSE-Z input deck.

## License and acknowledgements

BSD 2-Clause, see [LICENSE](LICENSE).

`zkit` includes a pure-Python implementation of the `igakit.igalib.bsp`
B-spline evaluator interface for wavefunction reconstruction; the upstream
BSD license is in `src/igakit/LICENSE.rst`.

<p align="center">
  <strong>Zakaria Dahbi</strong><br>
  King's College London · zdahbi@outlook.es
</p>

<p align="center">
  Z.D. acknowledges funding from UK Research and Innovation (UKRI) under the UK
  government's Horizon Europe funding guarantee (Grant No. EP/Z000807/1).
</p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/9c/UKRI_EPSR_Council-Logo_Horiz-RGB.png" alt="UKRI Engineering and Physical Sciences Research Council" width="220" />
</p>



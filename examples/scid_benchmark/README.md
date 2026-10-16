# TDSE-Z vs SCID-TDSE benchmark

A one-to-one comparison of TDSE-Z (3D Cartesian B-splines) with
[SCID-TDSE](https://github.com/MBI-Theory/scid-tdse) (spherical coordinates), the
standard spherical TDSE code in the strong-field literature. Both runs are
converged, and both include the infinite-time t-SURF(F) end term.

## Physics

| | |
|---|---|
| Target | H(1s), Coulomb tail switched off between r = 5 and 10 (SCID-TDSE's Becke-3 mask) |
| Pulse | ω = 0.4, E₀ = 0.05, 6-cycle sin² vector potential along z (SCID `z Sin2`), T = 174.25 |
| Photoelectrons | t-SURF(F) at R = 11.04, then the end term (the wave function left inside R at the end of the pulse) |

Because of the mask, the electron is free beyond the surface, so the Volkov states are exact in both codes.

## Runs

| | TDSE-Z (`tdsez/`) | SCID-TDSE (`scid/`) |
|---|---|---|
| Grid | 3D, ±22.5, hydrogenic knots (R1 0.1, NLin 4, NExp 10, HMax 1.3), 132 651 DOFs | l ≤ 30, m = 0, 540 log-uniform radial points (dr 0.07) |
| Propagator | Padé(2,2), dt = 0.1, `-tdsez_mat_free` | dt = 0.0025 |
| Absorber | Manolopoulos CAP, k_min = 0.6 | Manolopoulos CAP, k_min = 0.6 |
| t-SURF | `TSurff = 1`, `SurffEndTerm = 1`, (k_x, k_z) plane 151 × 301 | `sts_volkov`, `sts_volkov_atend`, 150 × 37 (k, θ) |
| Command | `mpirun -np 4 tdsez -inp quick.inp -tdsez_mat_free` | `spherical_tdse.x < quick.inp` (4 OpenMP threads) |
| Wall time | 16 min 48 s (end term 228 s) | 94 s |

The run logs are `tdsez/quick.out` and `scid/quick.out.gz`.

The data files are:

* **TDSE-Z:** `td/surff_quick.inp.h5`, `td/TimeEvolutionData_quick.inp.h5` and `static/EigenData_quick.inp.h5`. The wave-function file is not included.
* **SCID-TDSE:** `detail.table.gz` (norm, energy, dipole, acceleration and field) and `pes_volkov.table.gz`, both gzipped. The 28 MB `field.table` is left out because `detail.table` carries the same field.

## Reproduce

```bash
pip install "zkit-lib[viz]"
python reproduce.py
```

Or use the CLI:

```bash
zkit --run-dir tdsez compare-scid quick.inp \
    --scid-detail scid/detail.table.gz --scid-pes scid/pes_volkov.table.gz \
    --omega 0.4 --e0 0.05 --regions 0:0.12,0.12:0.47,0.47:0.9,0.9:1.2 \
    --outdir figures
```

## Results (`metrics.txt`)

| Quantity | TDSE-Z / SCID-TDSE |
|---|---|
| Field E_z(t) | identical (1.5 × 10⁻¹²) |
| Dipole ⟨z⟩(t), relative L2 difference | 5.4 × 10⁻³ |
| Harmonic spectrum (from the dipole), H0.5–H5.5 | median 0.010 decades |
| Total photoelectron yield | 1.032 |
| E < 0.12 (below the CAP k_min) | 1.155 |
| Two-photon order (0.12–0.47) | 1.018 |
| Three-photon region (0.47–0.9) | 1.007 |
| E > 0.9 | 0.978 |
| First ATI peak, point by point (dP/dE > 10 % of its maximum) | 0.884–1.150 |

The only systematic difference is below E = 0.12, where the CAP (k_min = 0.6) starts to reflect slow electrons.

The figures in `figures/` are:

* `scid_time.png`: field, dipole and acceleration;
* `scid_harmonics.png`: harmonic spectrum;
* `scid_pes.png`: dP/dE and its ratio;
* `scid_angular.png`: angular distributions at the ATI peaks;
* `scid_kmap.png`: momentum maps.

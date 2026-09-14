# Migration blueprint

The repository now uses the standard `src/` package layout. Generated files are
excluded from version control, Sphinx sources live in `docs/`, and GitHub Actions
builds the package, checks formatting and imports, tests Python 3.10 through 3.13,
and deploys documentation from `main`.

## Target structure

```text
zkit/
├── .github/workflows/
│   ├── ci.yml
│   └── docs.yml
├── docs/
│   ├── _static/logo.png
│   ├── examples/
│   ├── io/
│   ├── mwigner/
│   ├── physics/
│   ├── simulation/
│   ├── viz/
│   ├── Makefile
│   ├── conf.py
│   └── index.rst
├── examples/
│   ├── inputs/harmonic_oscillator_1d.inp
│   ├── inputs/harmonic_oscillator_2d.inp
│   ├── inputs/finite_quantum_well_heterostructure_1d.inp
│   ├── inputs/ho1d_rabi.inp
│   └── run_quick_examples.py
├── src/zkit/
│   ├── io/
│   ├── mwigner/
│   ├── viz/
│   ├── __init__.py
│   ├── __main__.py
│   ├── _version.py
│   ├── cli.py
│   ├── layouts.py
│   ├── models.py
│   ├── physics.py
│   ├── simulation.py
│   └── utils.py
├── tests/
│   ├── data/test_deck.inp
│   └── test_core.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Equivalent migration commands

Run these from the old repository root after committing or backing up local work:

```bash
mkdir -p src/zkit tests/data .github/workflows
mv __init__.py __main__.py _version.py cli.py layouts.py models.py \
  physics.py simulation.py utils.py src/zkit/
mv io mwigner viz src/zkit/
git rm -r docs/html
rmdir docs
mv docs_sphinx docs
mkdir -p docs/_static
mv logo/logo.png docs/_static/logo.png
rmdir logo
mv _test_deck.inp tests/data/test_deck.inp
rm -rf build UNKNOWN.egg-info zkit_lib.egg-info __pycache__ \
  src/zkit/io/__pycache__ src/zkit/mwigner/__pycache__ \
  src/zkit/viz/__pycache__ docs/_build
rm -f docs/_build.log MANIFEST.in
```

The committed `pyproject.toml`, `.gitignore`, Sphinx configuration, tests, and
workflows complete the migration after these filesystem operations.

## GitHub Pages setup

1. Push this tree to the `main` branch.
2. Open **Settings → Pages** in the GitHub repository.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Open the **Actions** tab and confirm the Documentation workflow succeeds.
5. The deployment job exposes the final Pages URL; the configured project URL is
   `https://dahbiz.github.io/zkit/`.

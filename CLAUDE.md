# CLAUDE.md

Guidance for Claude Code working in this repository. See `README.rst` for user-facing docs and https://anesthetic.readthedocs.io for full API documentation.

## Project overview

`anesthetic` is a post-processing library for nested sampling (and MCMC) chains. Its central design choice is that samples *are* `pandas.DataFrame`s: the public `Samples`, `MCMCSamples`, and `NestedSamples` classes subclass a `WeightedLabelledDataFrame`, which composes weighted-row support and a 2-level `(name, tex_label)` `MultiIndex` on columns. The matplotlib plotting backend is monkey-patched at import time (see `anesthetic/__init__.py`) so that `pandas` plotting routes through `anesthetic.plotting._matplotlib`. Statistics (`logZ`, `D_KL`, `d_G`, `logL_P`) and tension metrics are computed on these frames; plotting (`plot_1d`, `plot_2d`, `make_2d_axes`) uses weighted KDE/histogram/scatter kinds.

## Repo layout

- `anesthetic/` — package
  - `samples.py` — `Samples`, `MCMCSamples`, `NestedSamples`, `merge_nested_samples`, `stats`, `logZ`, `D_KL`. Large (~1500 lines); the heart of the library.
  - `weighted_pandas.py`, `labelled_pandas.py`, `weighted_labelled_pandas.py` — DataFrame/Series subclass machinery (weights, tex labels, MultiIndex columns). Touch with care: pandas internals leak through.
  - `plot.py` — `make_1d_axes`, `make_2d_axes`, `AxesSeries`, `AxesDataFrame`, plot kinds.
  - `plotting/_matplotlib/` — the registered pandas plotting backend.
  - `kde.py`, `boundary.py` — weighted KDE with boundary correction (recent active area; see commits 64f2b59, 021a03d, fa5bb4e).
  - `read/` — chain readers: `polychord`, `multinest`, `ultranest`, `nestedfit`, `cobaya`, `getdist`, `csv`, `hdf`. `read_chains(root)` auto-detects format.
  - `gui/` — interactive matplotlib widget for replaying nested runs (entry point `anesthetic` script).
  - `tension.py`, `convert.py`, `utils.py`, `testing.py`, `examples/perfect_ns.py`.
- `tests/` — pytest suite, `tests/example_data/` has small PolyChord/MultiNest/etc. fixtures used throughout.
- `docs/source/` — Sphinx (RTD theme, numpydoc, sphinx-autodoc-typehints).
- `bin/` — helper scripts: `run_tests`, `min_dependencies.py`, `latest_dependencies.py`, `check_up_to_date.py`, `bump_version.py`.

## Dev workflow

Install in editable mode with test extras:

    python -m pip install -e ".[test,all]"

Run the standard checks (mirrors `bin/run_tests` and CI in `.github/workflows/CI.yaml`):

    python -m flake8 anesthetic tests
    python -m pydocstyle --convention=numpy anesthetic
    python -m pytest

For headless / OSX matplotlib: `export MPLBACKEND=Agg` before pytest.

Build docs locally:

    python -m pip install -e ".[all,docs]"
    make -C docs html

Regenerate autodoc RSTs (only when adding modules):

    sphinx-apidoc -fM -t docs/templates/ -o docs/source/ anesthetic/

Pre-commit hooks (`.pre-commit-config.yaml`) run flake8 + pydocstyle. Install with `pre-commit install`.

## CI matrix (what your PR must pass)

CI runs on push and PR to `master` (`.github/workflows/CI.yaml`):
- `lint`: flake8 + pydocstyle, plus a grep check that every test file using `matplotlib` defines a `close_figures_on_teardown` fixture (see below).
- `sphinx`: `make html SPHINXOPTS="-W --keep-going -n"` — warnings are errors.
- `pip`: Python 3.10–3.14 on ubuntu, with and without `[all]` extras; plus macOS/Windows on 3.11.
- `conda`: same Python matrix on conda-forge.
- `minimum-dependencies` / `latest-dependencies`: pinned floors and unpinned ceilings — keep `pyproject.toml` bounds honest.
- `check-for-new-versions`: nightly cron check via `bin/check_up_to_date.py`.

## Code style

- numpy-style docstrings, enforced by `pydocstyle --convention=numpy` over `anesthetic/` (not tests).
- flake8 default config (no project-level overrides); applies to `anesthetic/` and `tests/`.
- Public API surface is exported from `anesthetic/__init__.py` — keep it stable; deprecations raise informative `ValueError`s (see `Samples.__init__` `root=` handling) rather than silent fallthrough.
- Subclass-friendly pandas patterns: preserve `_metadata`, return the correct subclass from operations. Look at `weighted_pandas.py` for the idioms; `tests/test_pandas_consistency.py` enforces them.

## Testing conventions

- Every test module that touches matplotlib must define and use a module-scoped `close_figures_on_teardown` fixture (CI greps for it). Copy the pattern from `tests/test_plot.py`.
- Optional-dependency tests use the skip/xfail helpers in `tests/utils.py` (`skipif_no_astropy`, `skipif_no_fastkde`, `skipif_no_getdist`, `skipif_no_h5py`, `pytables_mark_*`). Use these instead of bare `pytest.importorskip` so the matrix without `[all]` extras runs the xfail branch.
- Example chain fixtures live under `tests/example_data/` (PolyChord `pc`, `pc_250`, GetDist, MultiNest, etc.). Prefer these over generating synthetic data.
- `anesthetic.testing.assert_frame_equal` extends pandas' version to check `_metadata` round-trips; use it whenever comparing `Samples`-family frames.
- For statistic-heavy tests, `anesthetic.examples.perfect_ns.gaussian` / `correlated_gaussian` / `wedding_cake` produce analytic-truth nested runs.

## Where to look for X

- New plot kind: register in `anesthetic/plotting/_matplotlib/` and add docstring/tests in `tests/test_plot.py`. KDE boundary handling is in `kde.py` + `boundary.py`.
- New chain reader: add `anesthetic/read/<format>.py` exposing `read_<format>(root, ...)`, wire it into `anesthetic/read/chain.py`'s auto-detection, add a tiny fixture under `tests/example_data/<format>/`, and extend `tests/test_reader.py`.
- Stats/evidence/tension changes: `samples.py` (`stats`, `logZ`, `D_KL`, `_priors_pred`) and `tension.py`. Validate against `perfect_ns` analytics in `tests/test_samples.py`.
- Pandas-subclass plumbing bugs: `weighted_pandas.py` / `labelled_pandas.py` / `weighted_labelled_pandas.py`, then `tests/test_pandas_consistency.py`.

## Common pitfalls

- Don't import matplotlib at module top-level in new test files without the teardown fixture — CI will fail in the lint job, not pytest.
- The plotting backend override in `__init__.py` runs unconditionally on import; if a test stubs `pandas.options.plotting.backend`, restore it.
- Column access is two-level: `samples['x0']` works, but the column is really `('x0', '$x_0$')`. When iterating columns, use `samples.columns.get_level_values(0)`.
- Weights are stored as a pandas Index level (`weights`), not as a column. Use `samples.get_weights()` / `.set_weights()` rather than poking at the index.
- `read_chains(root)` takes a *root* (no extension) — auto-detection probes for sibling files.
- Minimum-deps CI is strict; if you bump a dependency, update both the floor in `pyproject.toml` and verify `bin/min_dependencies.py` resolves.

## Release / version

Version lives in `anesthetic/_version.py` (single source, picked up by `pyproject.toml` via `tool.setuptools.dynamic`). `bin/bump_version.py` handles release bumps; `.github/workflows/version.yaml` enforces version increments on PRs touching the package.

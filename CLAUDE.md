# CLAUDE.md

Guidance for Claude Code working in this repository. See `README.rst` for user-facing docs and https://anesthetic.readthedocs.io for full API documentation.

## Project overview

`anesthetic` is a post-processing library for nested sampling (and MCMC) chains. Its central design choice is that samples *are* `pandas.DataFrame`s: the public `Samples`, `MCMCSamples`, and `NestedSamples` classes subclass a `WeightedLabelledDataFrame`, which composes weighted-row support and a labelled-column machinery. When labels (often TeX) are supplied, columns become a 2-level `(name, label)` `MultiIndex`; otherwise they remain a plain Index (see `WeightedLabelledDataFrame.__init__` in `weighted_labelled_pandas.py`). The matplotlib plotting backend is monkey-patched at import time (`anesthetic/__init__.py`) so that `pandas` plotting routes through `anesthetic.plotting._matplotlib`; the same module also swaps in a custom `DataFrameFormatter` and sets `pandas.options.display.max_colwidth = 14`.

## Repo layout

- `anesthetic/` — package
  - `samples.py` — `Samples`, `MCMCSamples`, `NestedSamples`, `merge_nested_samples`, plus the stats methods `stats`, `logZ`, `D_KL`, `d_G`, `logL_P`, and the internals `logX`, `logdX`, `logw`, `_betalogL` (see roughly lines 781–1165). Large (~1500 lines); the heart of the library.
  - `weighted_pandas.py`, `labelled_pandas.py`, `weighted_labelled_pandas.py` — DataFrame/Series subclass machinery (weights, labels (often TeX), optional MultiIndex columns). Touch with care: pandas internals leak through.
  - `plot.py` — `make_1d_axes`, `make_2d_axes`, `AxesSeries`, `AxesDataFrame`, plot kinds. The local-linear boundary correction lives in `boundary.py` (used by the scipy `gaussian_kde` paths here).
  - `plotting/_matplotlib/` — the registered pandas plotting backend. `plotting/_core.py` defines the `PlotAccessor` and the `_common_kinds` / `_series_kinds` / `_dataframe_kinds` lists that gate `kind=` strings.
  - `kde.py` — fastKDE wrapper with reflection-based boundary handling (`mirror_1d`, `mirror_2d`). `boundary.py` — local-linear boundary correction for scipy KDE.
  - `read/` — chain readers: `polychord`, `multinest`, `ultranest`, `nestedfit`, `cobaya`, `getdist`, `csv`. `read_chains(root)` auto-detects between these. HDF5 round-trip is a separate top-level `read_hdf` / `to_hdf` pair (`anesthetic/read/hdf.py`), not part of the auto-detect chain.
  - `gui/` — interactive matplotlib widget for replaying nested runs (entry point `anesthetic` script).
  - `tension.py`, `convert.py`, `utils.py`, `testing.py`, `examples/perfect_ns.py`.
- `tests/` — pytest suite, `tests/example_data/` has small PolyChord/MultiNest/etc. fixtures used throughout.
- `docs/source/` — Sphinx (RTD theme, numpydoc, sphinx-autodoc-typehints).
- `bin/` — helper scripts: `run_tests`, `min_dependencies.py`, `latest_dependencies.py`, `check_up_to_date.py`, `bump_version.py`.

## Dev workflow

Install in editable mode with test extras:

    python -m pip install -e ".[test,all]"

Run the standard checks (roughly matches `bin/run_tests`; CI itself runs `flake8 anesthetic tests` — see `.github/workflows/CI.yaml`):

    python -m flake8 anesthetic anesthetic/gui tests
    python -m pydocstyle --convention=numpy anesthetic
    python -m pytest

For headless / OSX matplotlib: `export MPLBACKEND=Agg` before pytest.

Build docs locally:

    python -m pip install -e ".[all,docs]"
    make -C docs html

Regenerate autodoc RSTs (only when adding modules):

    sphinx-apidoc -fM -t docs/templates/ -o docs/source/ anesthetic/

Pre-commit hooks (`.pre-commit-config.yaml`) run flake8 on changed Python files under `anesthetic/` and `tests/`, and pydocstyle on changed Python files under `anesthetic/`. Install with `pre-commit install`.

## CI matrix (what your PR must pass)

CI runs on push and PR to `master`, and on a nightly schedule (`0 0 * * *`) (`.github/workflows/CI.yaml`):
- `lint`: flake8 (`anesthetic tests`) + pydocstyle (`--convention=numpy anesthetic`), plus a grep step intended to flag `tests/test*.py` files using `matplotlib` without a `close_figures_on_teardown` fixture — but the bash conditional in `CI.yaml:28` is malformed (`-ne 0]`, no space before `]`), so the check is a silent no-op. Add the fixture anyway.
- `sphinx`: `make html SPHINXOPTS="-W --keep-going -n"` — warnings are errors.
- `pip`: Python 3.10–3.14 on ubuntu, with and without `[all]` extras; plus macOS/Windows on 3.11.
- `conda`: same Python matrix on conda-forge.
- `minimum-dependencies` / `latest-dependencies`: pinned floors and unpinned ceilings — keep `pyproject.toml` bounds (`requires-python = ">=3.10"`, the authoritative source) honest.
- `check-for-new-versions`: runs `bin/check_up_to_date.py` as part of every CI invocation (the whole workflow runs on push, PR, and nightly schedule — the nightly run is the main use of this job).

## Code style

- numpy-style docstrings, enforced by `pydocstyle --convention=numpy` over `anesthetic/` (not tests).
- flake8 default config (no project-level overrides); applies to `anesthetic/` and `tests/`.
- Public API surface is exported from `anesthetic/__init__.py` — keep it stable. Legacy/removed behaviour signals itself with a mix of `ValueError` (e.g. `Samples.__init__` `root=`), `KeyError` (e.g. `read_chains(..., burn_in=...)`), `NotImplementedError` (legacy methods around `samples.py:447-456, 770-779, 942-947, 1096-1132`), and `warnings.warn` (e.g. `samples.py:161-173, 329-343`). Match the nearby pattern rather than picking one uniformly.
- Subclass-friendly pandas patterns: preserve `_metadata`, return the correct subclass from operations. Look at `weighted_pandas.py` for the idioms; the subclass/constructor behaviour is exercised by `tests/test_weighted_pandas.py`, `tests/test_labelled_pandas.py`, `tests/test_weighted_labelled_pandas.py`, and `tests/test_samples.py`. (`tests/test_pandas_consistency.py` is separate — it checks weighted statistic consistency against pandas for `mean`/`var`/`std`/`cov` etc.)

## Testing conventions

- Most test modules that touch matplotlib define a `close_figures_on_teardown` fixture. CI intends to grep for it, but the conditional in `CI.yaml:28` is malformed (`-ne 0]`) so the check is currently a silent no-op — don't rely on CI to catch a missing fixture. The existing examples (`tests/test_plot.py:28`, `tests/test_reader.py:23`, `tests/test_samples.py:27`) use the default *function* scope with `autouse=True`; copy that pattern. `tests/test_boundary.py` is a historical exception; add the fixture for any new test file regardless.
- Optional-dependency tests use the skip/xfail helpers in `tests/utils.py` (`skipif_no_astropy`, `skipif_no_fastkde`, `skipif_no_getdist`, `skipif_no_h5py`, `pytables_mark_*`). The `skipif_no_*` helpers skip parametrized cases; the `*_mark_xfail` variants mark whole tests as expected-fail. Pick whichever matches the nearby usage rather than bare `pytest.importorskip`.
- Example chain fixtures live under `tests/example_data/` (PolyChord `pc`, `pc_250`, GetDist, MultiNest, etc.). Prefer these over generating synthetic data.
- `anesthetic.testing.assert_frame_equal` extends pandas' version to check `_metadata` round-trips; use it whenever comparing `Samples`-family frames.
- For statistic-heavy tests, `anesthetic.examples.perfect_ns.gaussian` / `correlated_gaussian` / `wedding_cake` produce analytic-truth nested runs.

## Where to look for X

- New plot kind: register the class in the `PLOT_CLASSES` dict in `anesthetic/plotting/_matplotlib/__init__.py` (~lines 51–68) *and* extend the `_common_kinds` / `_series_kinds` / `_dataframe_kinds` tuples plus the accessor methods on `PlotAccessor` in `anesthetic/plotting/_core.py` (~lines 33–77). Add docstring/tests in `tests/test_plot.py`. KDE itself is in `kde.py` (fastKDE + reflection) and `boundary.py` (local-linear correction for scipy KDE).
- New chain reader: add `anesthetic/read/<format>.py` exposing `read_<format>(root, ...)`, wire it into the `readers` list in `anesthetic/read/chain.py` — *order matters*, the first reader whose `FileNotFoundError`/`IOError` does not fire wins, so a permissive new reader can shadow later ones. Add a tiny fixture under `tests/example_data/` matching the reader's expected file layout (existing readers use a mix of flat roots like `pc`, `mn`, `gd`, `cb` and subdirectories like `nf`, `un`, `mp`), and extend `tests/test_reader.py`.
- Stats/evidence/tension changes: `samples.py` (`stats`, `logZ`, `D_KL`, `d_G`, `logL_P`, plus `logX` / `logdX` / `logw` / `_betalogL`) and `tension.py`. Validate against `perfect_ns` analytics in `tests/test_samples.py`.
- Pandas-subclass plumbing bugs: `weighted_pandas.py` / `labelled_pandas.py` / `weighted_labelled_pandas.py`, then `tests/test_weighted_pandas.py`, `tests/test_labelled_pandas.py`, `tests/test_weighted_labelled_pandas.py`, and `tests/test_samples.py`. (`tests/test_pandas_consistency.py` covers weighted-vs-pandas statistic parity, not subclass plumbing.)

## Common pitfalls

- Don't import matplotlib at module top-level in new test files without the teardown fixture — the lint job is meant to catch this but currently doesn't (see "Testing conventions" above).
- The plotting backend override in `__init__.py` runs unconditionally on import (and also patches `pandas.io.formats.format.DataFrameFormatter`); if a test stubs `pandas.options.plotting.backend`, restore it.
- Column access may be two-level: when labels (often TeX) are present, `samples['x0']` works but the column is really `('x0', '$x_0$')`. When iterating columns, use `samples.columns.get_level_values(0)`.
- Weights are stored as a pandas Index level (`weights`), not as a column. Use `samples.get_weights()` / `.set_weights()` rather than poking at the index.
- `read_chains(root)` normally takes a *root* (no extension) and probes for sibling files; the CSV reader is the exception — it accepts a `.csv` path explicitly (see `anesthetic/read/csv.py`).
- HDF5 round-trip needs `anesthetic.read_hdf` (which restores `_metadata`), not `pandas.read_hdf`.
- Minimum-deps CI is strict; if you bump a dependency, update both the floor in `pyproject.toml` and verify `bin/min_dependencies.py` resolves.

## Release / version

The package version is read from `anesthetic/_version.py` (picked up by `pyproject.toml` via `tool.setuptools.dynamic`). The `:Version:` line in `README.rst` must stay in sync — `bin/check_version.py` enforces this, and `.github/workflows/version.yaml` runs it on PRs to `master`. Use `bin/bump_version.py` for release bumps; it updates both files.

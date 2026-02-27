# Agent Guide for pega-datascientist-tools

This repo is a Python library and tooling suite for Pega data science.
Primary code lives under `python/pdstools`, with tests in `python/tests`.

Keep edits focused, run the narrowest tests you can, and prefer `uv` for
dependencies and execution.

## Quick orientation
- `python/pdstools`: library code (polars-based data tooling, reports).
- `python/tests`: pytest suite.
- `python/docs`: Sphinx docs (with a Makefile).
- `reports/*.qmd`: Quarto report templates.
- `examples/`: notebooks and example content.

## Setup (local)
Use uv for env management, mirroring CI.

```bash
uv sync --extra tests
```

Optional extras by area:
- Healthcheck + reports: `uv sync --extra healthcheck --extra tests`
- Docs: `uv sync --extra docs --extra all`
- Pre-commit hooks: `uv sync --extra dev`

## Build / Lint / Test commands

### Linting and formatting
Preferred: run the pre-commit hooks (uses ruff + ruff-format + nb-clean).

```bash
uv run pre-commit run --all-files
```

If you want to run ruff directly:

```bash
uv run ruff check ./python
uv run ruff format ./python
```

### Tests
Full test suite (CI default, with coverage; skips heavy tests):

```bash
uv run pytest \
  --cov=./python/pdstools \
  --cov-report=xml \
  --cov-config=./python/tests/.coveragerc \
  --ignore=python/tests/test_healthcheck.py \
  --ignore=python/tests/test_ADMTrees.py
```

Run all tests locally:

```bash
uv run pytest python/tests
```

Run a single test file:

```bash
uv run pytest python/tests/test_Reports.py
```

Run a single test function:

```bash
uv run pytest python/tests/test_Reports.py::test_html_deduplication
```

Run by keyword:

```bash
uv run pytest python/tests -k "healthcheck"
```

Healthcheck tests (requires Quarto + Pandoc):

```bash
uv run pytest python/tests/test_healthcheck.py
```

### Docs
Docs live in `python/docs` and use Sphinx.

```bash
cd python/docs && make html
```

### Package build (release workflow)

```bash
python -m build --sdist --wheel --outdir dist/ .
```

## Code style guidelines

### Imports
- Order: stdlib, third-party, local (`pdstools...`).
- Prefer explicit imports; avoid wildcard imports.
- Keep imports at top of file; use `# noqa: F401` for intentional re-exports.

### Formatting
- Use ruff-format (black-compatible). Do not hand-format.
- Keep line length conservative; let the formatter decide.
- Use trailing commas in multi-line literals and call arguments.

### Types and typing
- Use type hints for public APIs and complex internal functions.
- Reuse aliases from `python/pdstools/utils/types.py` where suitable.
- Prefer `list[str]`, `dict[str, ...]`, and `|` unions (Py3.10+ syntax).

### Naming conventions
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: prefix with `_`.

### Error handling
- Raise specific exceptions; provide actionable messages.
- For optional dependencies, use `MissingDependenciesException` patterns.
- Prefer `ValueError`/`TypeError` for argument validation.
- Use `pytest.skip` for environment-dependent tests (e.g., Quarto).

### Logging
- Use module-level `logger = logging.getLogger(__name__)`.
- Avoid printing in library code; log at `debug`/`info` levels.

### Data and polars
- Use `polars.LazyFrame` where feasible; collect only at boundaries.
- Prefer expression-based transforms (`pl.Expr`) over Python loops.
- Keep IO in dedicated helpers; avoid side effects in pure functions.
- Column names in DecisionAnalyzer use human-friendly display names
  (e.g. "Issue", "Action", "Interaction ID") defined in
  `table_definition.py`. Never use internal Pega names (`pyIssue`,
  `pxInteractionID`, etc.) in new code. Internal working columns
  (`pxRank`, `is_mandatory`, `day`) are exceptions.

### Tests
- Use pytest fixtures for shared setup.
- Keep tests deterministic and data-driven.
- Mark slow tests with `@pytest.mark.slow`.
- Use `pytest.skip` for missing external tools (Quarto, Pandoc).

### Notebooks and reports
- Notebooks for docs should be empty/not pre-run.
- Quarto reports live in `reports/*.qmd`; keep params in YAML helpers.
- When changing reports, check size reduction behavior and Plotly embedding.
- Editing `.ipynb` notebook files directly is fine — modern tooling handles
  the JSON format well.

## Streamlit apps

### Architecture
- Apps live under `python/pdstools/app/<app_name>/` with a `Home.py`
  entry point and numbered pages in a `pages/` subdirectory.
- Three apps exist: `health_check`, `decision_analyzer`, `impact_analyzer`.
  All are launched via the unified CLI in `python/pdstools/cli.py`.
- **Shared utilities** go in `python/pdstools/utils/streamlit_utils.py`.
  App-specific helpers (data loading, custom widgets) go in a co-located
  `<app>_streamlit_utils.py` (e.g. `da_streamlit_utils.py`).

### Page boilerplate
Every page must start with two calls before any other Streamlit output:

```python
from pdstools.utils.streamlit_utils import standard_page_config
standard_page_config(page_title="Page Title · App Name")
```

Home pages additionally call `show_sidebar_branding(title)` to set the
sidebar logo and title (sub-pages re-apply it automatically).

### Session state conventions
- Store the main data object under a well-known key
  (e.g. `st.session_state.decision_data`).
- Guard every sub-page with an `ensure_data()` call that shows a
  warning and calls `st.stop()` when data is missing.
- Use the `_persist_widget_value` pattern (copy widget value to a
  second key on `on_change`) to survive Streamlit's page-navigation
  state clearing. Widget keys use a `_` prefix; persisted keys omit it.
- Prefix filter session-state keys with a `filter_type` string
  (`"global"`, `"local"`) so independent filter sets don't collide.

### Caching
- Use `@st.cache_resource` for stateful / non-serializable objects
  (e.g. `DecisionAnalyzer` instances). Prefix unhashable parameters
  with `_` and supply a `data_fingerprint` string for cache busting.
- Use `@st.cache_data` for pure data transforms and plot functions.
  When inputs include `pl.LazyFrame` or `pl.Expr`, pass a custom
  `hash_funcs` dict (see `polars_lazyframe_hashing` in
  `da_streamlit_utils.py`).

### CLI integration
- CLI flags (`--deploy-env`, `--data-path`, `--sample`, `--temp-dir`)
  are propagated to the Streamlit process as `PDSTOOLS_*` environment
  variables. Read them via helpers in `streamlit_utils.py`
  (`get_deploy_env()`, `get_data_path()`, etc.), never via
  `os.environ` directly in page code.

### Plotly charts
- All charts use Plotly. Display with `st.plotly_chart(...,
  use_container_width=True)`.
- Keep plot construction in the library layer (`plots.py`); Streamlit
  pages call those functions and may tweak layout via
  `.update_layout()` before rendering.

### Shared About page
- Use `show_about_page()` from `streamlit_utils.py` for a
  standardised About page. A page file can be as small as two lines
  (page config + `show_about_page()`).

### General Streamlit rules
- Never use `st.experimental_*` APIs — they have been removed. Use
  the stable equivalents (`st.cache_data`, `st.cache_resource`, etc.).
- Use Streamlit magic (bare strings/expressions) for static markdown
  where convenient, but prefer explicit `st.markdown()` / `st.write()`
  for dynamic content.
- Keep heavy computation out of page scripts; delegate to cached
  functions or the library layer.

## Contrib and workflow notes
- Main tests are `python/tests`; CI runs multi-OS and multi-Python.
- Healthcheck tests are separate and require extra deps/tools.
- Use `uv` to mirror CI; avoid mixing system pip unless necessary.

## Cursor/Copilot rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md`
  were found in this repository at the time of writing.

## Tips for agentic changes
- Prefer minimal diffs; avoid touching unrelated files.
- Do not edit generated files (e.g., `uv.lock`) unless required.
- Keep security in mind; do not commit secrets or local data artifacts.

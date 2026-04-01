# Agent Guide for pega-datascientist-tools

This repo is a Python library and tooling suite for Pega data science.
Primary code lives under `python/pdstools`, with tests in `python/tests`.

Keep edits focused, run the narrowest tests you can, and prefer `uv` for
dependencies and execution.

## Critical rules

- **Open source repo.** Never include customer names, customer data, or
  internal project names anywhere (code, comments, commit messages, tests).
- **Git workflow.** Do not `git commit` on the user's working branch.
  Stage with `git add` and let the user commit. Exception: branches you
  created yourself, or when the user explicitly asks for a commit/PR.
- **Product naming.** User-facing text: "Decision Analysis Tool",
  "ADM Health Check". Code/CLI: `decision_analyzer`, `adm_healthcheck`.
- **No `verbose` parameters.** Use `logging.debug()` instead of
  `if verbose: print(...)`. Remove `verbose` if you encounter it.
- **Prefer Polars over Pandas** for all data processing.

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
- **Optional dependencies**: use lazy imports inside the method that needs
  them (see `local_model_utils.py` for the pattern). Do not use
  module-level `try/except ImportError` blocks.

### Formatting
- Use ruff-format (black-compatible). Do not hand-format.
- Keep line length conservative; let the formatter decide.
- Use trailing commas in multi-line literals and call arguments.

### Types and typing
- Use type hints for public APIs and complex internal functions.
- Reuse aliases from `python/pdstools/utils/types.py` where suitable.
- Python 3.10+: use `list[str]`, `dict[str, ...]`, `X | None` (not
  `Optional[X]`). Do not import `Optional`, `Union`, `List`, `Dict`
  from `typing`.

### Naming conventions
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: prefix with `_`.
- **Don't leak external naming into Python APIs.** When a downstream
  system uses a different naming scheme (e.g. Pega's `pyCreatedBy`),
  keep the Python-facing field name Pythonic (`created_by`) and handle
  the translation in the serialization layer — use Pydantic
  `serialization_alias` / `alias`, a custom encoder, or a `to_json()`
  method.

### Error handling
- Raise specific exceptions; provide actionable messages.
- For optional dependencies, use `MissingDependenciesException` patterns.
- Prefer `ValueError`/`TypeError` for argument validation.
- Use `pytest.skip` for environment-dependent tests (e.g., Quarto).
- **Package-manager neutral messages**: do not hardcode `pip install`
  or `uv pip install` in error text. Name the missing package and let
  the user choose their tool.

### Logging
- Use module-level `logger = logging.getLogger(__name__)`.
- No `print()` in library code; log at `debug`/`info` levels.
- `debug` as a parameter name: only when it changes the **return value**
  (extra columns, etc.), not for controlling log output.

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
- **Minimum coverage: 80 %** for new and overall code (CI-enforced).
- **Strongly prefer exact-value assertions** over structural checks.
  Tests that only verify column names, non-empty results, or `height > 0`
  give false confidence — they pass even when computations are wrong.
  Use small, minimal datasets (see `data/da/sample_eev2_minimal.csv`)
  where every expected value can be traced back to the input data.
  Structural/smoke tests on larger datasets are fine as a complement,
  but the correctness backbone should be exact-value tests.

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
- All charts use Plotly. Display with `st.plotly_chart(fig)` — do NOT
  pass `use_container_width` (deprecated) or `width="stretch"` (default).
  Use `config=` for Plotly config options.
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
- Avoid `components.html()` with custom JavaScript. It runs in an
  iframe, is fragile, hard to debug, and does not integrate with
  Streamlit's state management (session state, theming, reruns).
  Prefer native Streamlit widgets and layout primitives. Only resort
  to custom JS when there is genuinely no Streamlit-native alternative.
- Use Streamlit magic (bare strings/expressions) for static markdown
  where convenient, but prefer explicit `st.markdown()` / `st.write()`
  for dynamic content.
- Keep heavy computation out of page scripts; delegate to cached
  functions or the library layer.

## Design principles for new functionality

These principles apply when adding new features, modules, or classes —
whether in a PR or an agentic edit.

### Extend before you create
Before adding a new class or module, search the codebase for existing
models or utilities that already cover similar ground. Add methods,
fields, or class-methods to the existing code rather than building a
parallel hierarchy. Duplicate abstractions create maintenance burden and
confuse users about which entry point to use.

### Earn your abstractions
Don't introduce design patterns (Builder, Factory, Registry, etc.)
unless the complexity genuinely demands them. If a Pydantic model gives
you validation, defaults, and serialization for free, use its
constructor directly. A fluent builder that only assigns fields and
calls a constructor adds indirection without benefit.

### One concern per PR
Keep pull requests focused on a single, well-defined change. If a
feature touches multiple independent areas (e.g. conversion + security +
optimization), split them into separate PRs so each can be reviewed and
merged on its own merits.

### Right-size your validation and security
Validate inputs and enforce sensible limits (allow-lists, size caps,
type checks). But don't add threat mitigations that don't map to a real
attack surface — e.g. scanning for SQL injection in values that are
never executed as SQL, or HTML-sanitising strings that are never
rendered in a browser. Keep security code proportional to the actual
risk; otherwise it creates false confidence and false-positive noise.

### Prefer thin, valuable wrappers
When wrapping a third-party library, add genuine value: compatibility
fixes, sane defaults, validation, or integration with pdstools
conventions. Don't wrap standard calls with layers of config objects
and result models just for the sake of wrapping — if calling the
library directly is clear enough, let users do that.

### Keep the Python API Pythonic
Public APIs should feel natural to a Python developer. Translate
between external naming conventions (Pega property names, JSON key
schemes) at the serialization boundary, not in field names or
function signatures. See the naming-conventions section above.

## Feature backlog / TODO files

Major features maintain a living TODO file in `docs/plans/` (e.g.
`decision-analyzer-TODO.md`, `health-check-TODO.md`,
`impact-analyzer-TODO.md`). These track open work items, bugs, and
improvement ideas that are not otherwise captured in GitHub issues.

- **Check before working.** When starting work on a feature area, read
  its TODO file first for context on known issues and planned work.
- **Update as you go.** Mark items done (`[x]`) when they land on
  master. Add new items when you discover bugs, limitations, or ideas
  during development.
- **Priority levels:** P1 = high, P2 = medium, P3 = nice-to-have.
- **Keep them current.** Remove or archive completed sections
  periodically so the files stay useful.

## Contrib and workflow notes
- Main tests are `python/tests`; CI runs multi-OS and multi-Python.
- Healthcheck tests are separate and require extra deps/tools.
- Use `uv` to mirror CI; avoid mixing system pip unless necessary.

## Tips for agentic changes
- Prefer minimal diffs; avoid touching unrelated files.
- Do not edit generated files (e.g., `uv.lock`) unless required.
- Keep security in mind; do not commit secrets or local data artifacts.
- When the IDE shows type errors after an edit, distinguish
  **pre-existing** from **newly introduced**. Only fix the new ones
  silently; ask before touching pre-existing issues.

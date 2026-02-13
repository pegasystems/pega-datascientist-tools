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

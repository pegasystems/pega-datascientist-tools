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
- **Stay focused, but fix tightly-coupled bugs.** Don't fix
  pre-existing issues unrelated to the task at hand. However, if a
  refactor surfaces a bug that is directly caused by or tightly
  coupled to the code you're changing (e.g. a wrong return-type
  annotation that callers were defensively working around, a stale
  `# type: ignore` that hid a real mismatch), fix it in the same PR
  and call it out in the commit message. Splitting it off creates a
  regression window where the fix lacks the surrounding context.
- **Product naming.** User-facing text: "Decision Analysis Tool",
  "ADM Health Check". Code/CLI: `decision_analyzer`, `adm_healthcheck`.
- **`verbose` parameters.** Reserve `verbose` for genuine user-facing
  progress — tqdm progress bars on long-running I/O, multi-stage CLI
  output (`"Step 1/3: writing parquet…"`). For anything that prints
  internal technical detail (subprocess output, decoded values, "file
  not found in dir X", schema warnings), drop the parameter and use
  `logger.debug()` / `logger.info()` instead. Never overload `verbose`
  to control non-output behaviour (e.g. cache invalidation, retry
  policy) — that's a separate parameter.
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
The full suite takes ~10–15 minutes locally. **Prefer narrow, targeted runs**
(single file, single function, or `-k` keyword filter) while iterating.
Only run the full suite when you're done with a logical chunk of work
or when you need broad confidence (e.g. before pushing a PR). CI re-runs
everything, so a final full local run is usually enough.

Full test suite (CI default, with coverage; skips heavy tests):

```bash
uv run pytest \
  --cov=./python/pdstools \
  --cov-report=xml \
  --cov-config=./python/tests/.coveragerc \
  --ignore=python/tests/test_healthcheck.py \
  --ignore=python/tests/test_explanations_report.py \
  --ignore=python/tests/test_batch_healthcheck.py \
  --ignore=python/tests/explanations \
  -n auto
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

#### `# type: ignore` is a smell, not a tool
Treat `# type: ignore` as a last resort, and treat existing ones as
suspicious during any refactor — they often hide real bugs.

- **Drop redundant explicit annotations first.** Most
  `# type: ignore[assignment]` on lines like
  `df: pl.DataFrame = lf.collect()` are caused by the redundant
  annotation itself (polars overloads return the right type already).
  Remove the annotation; the ignore disappears with it.
- **Use `cast` over `# type: ignore[arg-type]` for genuine narrowing.**
  `cast(list[str], scope_config["group_cols"])` documents intent;
  `# type: ignore[arg-type]` hides it.
- **Class-level annotations for lazily-set attributes.** When an
  attribute is assigned from a method (not `__init__`), declare it at
  class level (`_num_sample_interactions: int`) instead of writing
  `# type: ignore[attr-defined]` at every access site.
- **`# type: ignore[return-value]` almost always means the signature
  lies.** If you have to suppress a return mismatch, the real fix is
  to update the function's annotated return type to match what it
  actually returns. Callers downstream are probably defensively
  working around the wrong type.
- Reserve `# type: ignore` for genuinely external problems
  (third-party stubs missing, library bugs) and add a comment
  explaining why.

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
- **Drop `# pragma: no cover` when you add tests.** Pragmas are for
  code that genuinely can't be exercised (platform-specific branches,
  defensive `raise` after exhaustive `if/elif`). Once a method has a
  test, the pragma is stale and misleading — remove it in the same PR
  that adds the test.

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

### Keep parameter surfaces small
Every parameter on a public function is an API commitment. Only
expose what users genuinely need to control. Internal filtering or
behavior toggles should be handled close to where they act, not
threaded through every layer of a call chain. If a toggle only
matters at one point in the pipeline, apply it there — e.g. filter
in the data-loading step, or accept a pre-filtered frame — rather
than adding it to every function between the caller and that point.
**Smell:** the same parameter name appearing in 3+ functions in the
same call chain.

### Use Python's built-in defaults
Use keyword arguments with literal defaults
(`def foo(top_n: int = 20)`) — this is idiomatic Python,
self-documenting, and immediately visible in IDE tooltips. Avoid
centralizing defaults in a config dataclass or defaults singleton
that every function references; that adds indirection without
benefit for a library. The function signature *is* the
documentation of its defaults. A frozen-dataclass of defaults is
an engine/enterprise pattern, not a fit for a lightweight analysis
package.

### Namespace facade for large analyzer classes
For any class that grows beyond ~20 public methods, split related
methods into sub-namespace classes attached as instance attributes.
`ADMDatamart` is the reference: `dm.plot`, `dm.aggregates`, `dm.agb`,
`dm.generate`, `dm.bin_aggregator`. Each sub-namespace:

- Lives in its own module (`adm/Plots.py`, `adm/Aggregates.py`, …).
- Takes a single parent argument named **`datamart`** (or the
  equivalent name used by the parent class) and stores it as
  `self.datamart`.
- Uses `if TYPE_CHECKING: from .Parent import Parent` to avoid
  circular imports for the back-reference.
- Is instantiated in the parent's `__init__` and assigned to a
  short, dot-completion-friendly attribute (`self.plot = Plots(self)`).

This keeps the public API discoverable (one class to import; dot
completion reveals everything) without ballooning the parent into a
multi-thousand-line god class. Use the same pattern when refactoring
existing fat classes — don't invent a new convention.

### I/O lives in classmethods, not `__init__`
Keep `__init__` pure: it should only accept already-loaded data
structures (typically `pl.LazyFrame`s) and configuration. All file,
network, and S3 I/O lives in alternative constructors named
`from_<source>` (e.g. `from_ds_export`, `from_s3`,
`from_dataflow_export`, `from_pdc`). This mirrors the
`pl.read_csv` / `pl.scan_csv` idiom and makes the class trivially
testable with synthesized data — no monkey-patching required.

### `return_df` parameter on plot methods
Every public method that produces a chart should accept
`return_df: bool = False` as a keyword-only argument. When `True`,
return the underlying (Lazy)Frame that drives the chart instead of
the figure. This pattern (used consistently across `ADMDatamart.plot`)
makes plots scriptable, testable, and composable without forcing
users to re-derive the aggregation. Pair with `@overload` so type
checkers know which return shape applies.

## Feature backlog / TODO files

Major features maintain a living backlog in `docs/plans/`. Two layouts
are in use, with the per-item-file layout preferred for any new or
churning area:

### Per-item-file layout (preferred)

`docs/plans/<feature>/<slug>.md` — one Markdown file per open item,
plus a `README.md` describing the convention. The `adm/` backlog is the
reference example.

- **Adding an item** = create a new file. No shared state; no merge
  conflicts with parallel PRs.
- **Resolving an item** = `git rm` the file in the PR that resolves it.
  The deletion is the audit trail; the PR description / commit message
  captures the *what* and *why*.
- **Listing open items** = `ls docs/plans/<feature>/`.
- **Filename**: short kebab-case slug (`lazy-plotly.md`,
  `binagg-rename.md`).
- **Contents**: title, priority (P1/P2/P3), files touched, problem
  statement, proposed approach, any cross-refs. Aim for under one
  screen.

### Single-file layout (legacy)

Some areas (e.g. `decision-analyzer-TODO.md`, `health-check-TODO.md`,
`impact-analyzer-TODO.md`) still use a single Markdown file with Open
and Done sections. This works fine when the file sees little churn.
Convert to the per-item-file layout the next time the area sees
significant parallel PR activity (the recurring conflict on the Done
section is the trigger).

### Common rules (both layouts)

- **Check before working.** When starting work on a feature area, read
  its plan file/dir first.
- **Update as you go.** Mark items done (or `git rm` them) when they
  land on master. Add new items when you discover bugs, limitations, or
  ideas during development.
- **Priority levels:** P1 = high, P2 = medium, P3 = nice-to-have.

### Inline `# TODO` vs plan-file entries

Use the right venue for the right kind of note:

- **Plan file (`docs/plans/<feature>-TODO.md`)** — substantive backlog
  items: missing features, refactors that span more than a few lines,
  known limitations, bugs worth tracking, design questions. Anything a
  future contributor would want to discover by reading the backlog
  rather than by stumbling onto a comment.
- **Inline `# TODO`** — small, code-local hints tied to a specific line:
  "consider a faster path here", "revisit when polars supports X",
  "this branch is only hit in tests". Should be self-contained — no
  required cross-reference for a reader to act on it.
- **Anchor when both apply.** If an inline note has a meaningful
  backlog entry, link them with
  `# Tracked in docs/plans/<file>.md (P<n>: <item title>) — <one-liner>`
  so the call site stays searchable and the rationale is visible
  locally. Don't leave dangling inline TODOs that duplicate a backlog
  item without the anchor.
- **Don't park lists of TODOs at the top of a file/page.** Lift them
  into the relevant plan file and replace the block with a single
  pointer comment.

## Surfacing follow-up work

When a conversation or refactor uncovers something that won't be
addressed in the current PR, *don't let it evaporate*. Pick the right
venue and either file it or hand the user a draft they can file:

- **GitHub issue** for anything other contributors should see and pick
  up: bugs, missing features, design questions, promotion candidates
  from prototype → product. Agents typically can't `gh issue create`
  on this repo (EMU restrictions) — write a ready-to-paste draft
  (title + body in markdown) and hand it to the user. Keep one issue
  per concern, not omnibus dumps.
- **Plan-file entry** (`docs/plans/<feature>-TODO.md`) for backlog
  items scoped to a specific feature area that aren't substantial
  enough to be their own issue, or that need design work before they
  can be filed.
- **Inline `# TODO`** for code-local hints (see "Inline `# TODO` vs
  plan-file entries" below).

Triggers to raise follow-ups proactively:
- A "we should also…" or "while we're here…" thought that's out of
  scope for the current PR.
- A bug discovered that *isn't* tightly coupled to the current change
  (the coupled ones get fixed in-PR — see "Stay focused" in critical
  rules).
- A `# type: ignore`, defensive `isinstance` branch, or `# pragma: no
  cover` you can't justify removing in the current PR.
- Repeated workarounds across multiple files that point to a missing
  abstraction.
- Prototype code that has matured enough to belong in `pdstools` proper.

Default to nudging the user: *"This is out of scope for the current
PR — want me to draft an issue?"* — better one too many drafts than a
silently-dropped follow-up.

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

## Parallel sub-agent workflow (git worktrees)

When dispatching multiple background sub-agents that each work on a
**different branch in parallel**, they MUST be isolated via
`git worktree` — otherwise they share the user's working directory,
which is a recipe for:

- One agent's `git checkout <branch>` switching the user's HEAD without
  warning.
- Agent A's WIP edits showing up in agent B's `git status` and getting
  swept into the wrong commit.
- The user's foreground edits getting tangled with agent edits and
  ending up on the wrong branch.

### Setup

Create one worktree per parallel branch under a sibling directory:

```bash
mkdir -p ../pdstools-worktrees
git worktree add ../pdstools-worktrees/<branch-name> <branch-name>
```

Each sub-agent is told to `cd ../pdstools-worktrees/<branch-name>` first
and do **all** work there (edits, tests, commits, push). The agent prompt
should explicitly forbid touching the main checkout.

After `git worktree add`, run `uv sync --extra tests` once in the new
worktree — the `.venv` is per-worktree, not shared.

### Cleanup

When a branch is merged or abandoned, prune its worktree:

```bash
git worktree remove ../pdstools-worktrees/<branch-name>
```

### When to skip worktrees

- A single background sub-agent + the user staying in read-only mode in
  the main checkout is usually fine.
- Sequential sub-agent runs (one finishes before the next starts) don't
  need worktrees.
- The moment a second concurrent writer is involved, set up worktrees.

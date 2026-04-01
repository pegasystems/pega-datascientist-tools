# `--filter` Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--filter` CLI flag that extracts specific rows from large files by exact-match on user-friendly column names. Can be combined with `--sample` (filter applied first, then sampling).

**Architecture:** New `parse_filter_specs()` + `resolve_filter_column()` functions in `decision_analyzer/utils.py` handle parsing and column name resolution. CLI adds `--filter` arg, propagates via `PDSTOOLS_FILTER` env var. Home.py reads the env var, applies the filter as a lazy Polars expression, then caches via `prepare_and_save()`.

**Tech Stack:** Python, Polars (LazyFrame expressions), argparse, Streamlit

**Design doc:** `docs/plans/2026-04-01-filter-flag-design.md`

**Design change (during implementation):** `--filter` and `--sample` are NOT
mutually exclusive. Filter is applied first, then sampling on the filtered result.
Tasks 3 and 5 below were updated accordingly during implementation.

---

### Task 1: `resolve_filter_column()` -- column name resolution

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py` (add after `parse_sample_flag` ~line 975)
- Test: `python/tests/test_da_utils.py`

**Step 1: Write the failing tests**

Add to `test_da_utils.py`. Import `resolve_filter_column` at the top alongside the other imports.

```python
class TestResolveFilterColumn:
    """Tests for resolve_filter_column()."""

    def test_resolve_display_name_v2(self):
        """Display name 'Interaction ID' resolves to raw key 'pxInteractionID'."""
        result = resolve_filter_column(
            "Interaction ID", available_columns={"pxInteractionID", "pyIssue"}
        )
        assert result == "pxInteractionID"

    def test_resolve_display_name_v1(self):
        """Display name 'Subject ID' resolves to v1 raw key 'pySubjectID'."""
        result = resolve_filter_column(
            "Subject ID", available_columns={"pySubjectID", "pxInteractionID"}
        )
        assert result == "pySubjectID"

    def test_resolve_alias(self):
        """Alias 'InteractionID' resolves to 'pxInteractionID'."""
        result = resolve_filter_column(
            "InteractionID", available_columns={"pxInteractionID"}
        )
        assert result == "pxInteractionID"

    def test_resolve_raw_key_fallback(self):
        """Raw column name works as fallback."""
        result = resolve_filter_column(
            "pxInteractionID", available_columns={"pxInteractionID"}
        )
        assert result == "pxInteractionID"

    def test_resolve_case_insensitive(self):
        """Column name resolution is case-insensitive."""
        result = resolve_filter_column(
            "interaction id", available_columns={"pxInteractionID"}
        )
        assert result == "pxInteractionID"

    def test_resolve_channel_v2(self):
        """'Channel' resolves to v2 raw key when present."""
        result = resolve_filter_column(
            "Channel",
            available_columns={"Primary_ContainerPayload_Channel", "pyIssue"},
        )
        assert result == "Primary_ContainerPayload_Channel"

    def test_resolve_channel_already_renamed(self):
        """'Channel' resolves to itself when data already uses display names."""
        result = resolve_filter_column(
            "Channel", available_columns={"Channel", "Issue"}
        )
        assert result == "Channel"

    def test_resolve_unknown_column_raises(self):
        """Unknown column name raises ValueError with available columns listed."""
        with pytest.raises(ValueError, match="Unknown filter column 'Bogus'"):
            resolve_filter_column(
                "Bogus", available_columns={"pxInteractionID"}
            )
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_da_utils.py::TestResolveFilterColumn -v`
Expected: FAIL with `ImportError` (function does not exist yet)

**Step 3: Write minimal implementation**

Add to `python/pdstools/decision_analyzer/utils.py` after `parse_sample_flag` (~line 975):

```python
def resolve_filter_column(
    name: str,
    available_columns: set[str],
) -> str:
    """Resolve a user-friendly column name to the actual column in the data.

    Checks display names, aliases, and raw keys from both the
    ``DecisionAnalyzer`` and ``ExplainabilityExtract`` schemas.
    Resolution is case-insensitive.

    Parameters
    ----------
    name : str
        User-provided column name (display name, alias, or raw key).
    available_columns : set[str]
        Column names actually present in the raw data.

    Returns
    -------
    str
        The actual column name present in *available_columns*.

    Raises
    ------
    ValueError
        If *name* cannot be resolved to any column in *available_columns*.
    """
    name_lower = name.lower()

    # Check if name is directly present (raw key or already-renamed display name)
    for col in available_columns:
        if col.lower() == name_lower:
            return col

    # Build lookup: display_name/alias -> list of raw keys
    for schema in (DecisionAnalyzer, ExplainabilityExtract):
        for raw_key, config in schema.items():
            display = config["display_name"]
            aliases = config.get("aliases", [])
            all_names = [display] + aliases

            if any(n.lower() == name_lower for n in all_names):
                # Found a schema match -- check if raw_key is in the data
                if raw_key in available_columns:
                    return raw_key
                # Or if the display name itself is in the data (already renamed)
                if display in available_columns:
                    return display

    # Build helpful error message
    filterable = sorted(
        {
            config["display_name"]
            for schema in (DecisionAnalyzer, ExplainabilityExtract)
            for config in schema.values()
            if any(
                c in available_columns
                for c in [config["display_name"]]
                + config.get("aliases", [])
                + [
                    raw_key
                    for raw_key, cfg in schema.items()
                    if cfg is config
                ]
            )
        }
    )
    raise ValueError(
        f"Unknown filter column {name!r}. "
        f"Available filterable columns: {', '.join(filterable)}"
    )
```

Also add `resolve_filter_column` to the imports in `test_da_utils.py`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_da_utils.py::TestResolveFilterColumn -v`
Expected: all 8 tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/utils.py python/tests/test_da_utils.py
git commit -m "feat(filter): add resolve_filter_column() with tests"
```

---

### Task 2: `parse_filter_specs()` -- parse and build Polars expression

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py` (add after `resolve_filter_column`)
- Test: `python/tests/test_da_utils.py`

**Step 1: Write the failing tests**

```python
class TestParseFilterSpecs:
    """Tests for parse_filter_specs()."""

    def test_single_filter_single_value(self):
        """Single filter with one value produces is_in expression."""
        df = pl.LazyFrame({"pxInteractionID": ["A", "B", "C"], "x": [1, 2, 3]})
        expr = parse_filter_specs(
            ["Interaction ID=A"],
            available_columns={"pxInteractionID", "x"},
        )
        result = df.filter(expr).collect()
        assert result["pxInteractionID"].to_list() == ["A"]

    def test_single_filter_multiple_values(self):
        """Single filter with comma-separated values keeps matching rows."""
        df = pl.LazyFrame({"pxInteractionID": ["A", "B", "C"], "x": [1, 2, 3]})
        expr = parse_filter_specs(
            ["Interaction ID=A,C"],
            available_columns={"pxInteractionID", "x"},
        )
        result = df.filter(expr).collect()
        assert result["pxInteractionID"].to_list() == ["A", "C"]

    def test_multiple_filters_and(self):
        """Multiple filters are ANDed together."""
        df = pl.LazyFrame({
            "pxInteractionID": ["A", "B", "C"],
            "pyIssue": ["Sales", "Service", "Sales"],
        })
        expr = parse_filter_specs(
            ["Interaction ID=A,C", "Issue=Sales"],
            available_columns={"pxInteractionID", "pyIssue"},
        )
        result = df.filter(expr).collect()
        assert result["pxInteractionID"].to_list() == ["A"]
        assert result["pyIssue"].to_list() == ["Sales"]

    def test_no_matching_rows(self):
        """Filter that matches nothing produces empty result."""
        df = pl.LazyFrame({"pxInteractionID": ["A", "B"], "x": [1, 2]})
        expr = parse_filter_specs(
            ["Interaction ID=Z"],
            available_columns={"pxInteractionID", "x"},
        )
        result = df.filter(expr).collect()
        assert result.height == 0

    def test_malformed_spec_no_equals(self):
        """Spec without '=' raises ValueError."""
        with pytest.raises(ValueError, match="Expected format"):
            parse_filter_specs(
                ["Interaction ID"],
                available_columns={"pxInteractionID"},
            )

    def test_malformed_spec_empty_values(self):
        """Spec with empty values raises ValueError."""
        with pytest.raises(ValueError, match="Expected format"):
            parse_filter_specs(
                ["Interaction ID="],
                available_columns={"pxInteractionID"},
            )

    def test_values_are_case_sensitive(self):
        """Filter values are case-sensitive."""
        df = pl.LazyFrame({"pxInteractionID": ["ABC", "abc"], "x": [1, 2]})
        expr = parse_filter_specs(
            ["Interaction ID=ABC"],
            available_columns={"pxInteractionID", "x"},
        )
        result = df.filter(expr).collect()
        assert result["pxInteractionID"].to_list() == ["ABC"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_da_utils.py::TestParseFilterSpecs -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `python/pdstools/decision_analyzer/utils.py` after `resolve_filter_column`:

```python
def parse_filter_specs(
    filter_specs: list[str],
    available_columns: set[str],
) -> pl.Expr:
    """Parse ``--filter`` specs into a combined Polars filter expression.

    Each spec has the form ``"Column Name=value1,value2,..."``. Column names
    are resolved via :func:`resolve_filter_column` (case-insensitive). Values
    are exact-match and case-sensitive. Multiple specs are ANDed together.

    Parameters
    ----------
    filter_specs : list[str]
        Filter specifications from the CLI.
    available_columns : set[str]
        Column names present in the raw data.

    Returns
    -------
    pl.Expr
        Combined filter expression.

    Raises
    ------
    ValueError
        If a spec is malformed or references an unknown column.
    """
    if not filter_specs:
        raise ValueError("filter_specs must not be empty")

    expressions: list[pl.Expr] = []

    for spec in filter_specs:
        if "=" not in spec:
            raise ValueError(
                f"Expected format: 'Column Name=value1,value2', got {spec!r}"
            )
        col_name, values_str = spec.split("=", 1)
        col_name = col_name.strip()
        values_str = values_str.strip()

        if not values_str:
            raise ValueError(
                f"Expected format: 'Column Name=value1,value2', got {spec!r}"
            )

        values = [v.strip() for v in values_str.split(",")]
        resolved = resolve_filter_column(col_name, available_columns)
        expressions.append(pl.col(resolved).is_in(values))

    combined = expressions[0]
    for expr in expressions[1:]:
        combined = combined & expr
    return combined
```

Also add `parse_filter_specs` to the imports in `test_da_utils.py`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_da_utils.py::TestParseFilterSpecs -v`
Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/utils.py python/tests/test_da_utils.py
git commit -m "feat(filter): add parse_filter_specs() with tests"
```

---

### Task 3: CLI `--filter` argument and env var propagation

**Files:**
- Modify: `python/pdstools/cli.py:76-99` (add `--filter` arg after `--sample`)
- Modify: `python/pdstools/cli.py:236-256` (propagate to env var, validate exclusivity)
- Test: `python/tests/test_da_utils.py` (or a new CLI test if one exists)

**Step 1: Write the failing test**

Add a test that validates the argparse setup. Since the CLI launches Streamlit
(hard to test end-to-end), test the argument parsing and env var logic directly.

```python
class TestFilterCLIArgs:
    """Tests for --filter CLI argument parsing."""

    def test_filter_arg_parsed(self):
        """--filter args are collected into a list."""
        from pdstools.cli import build_parser

        parser = build_parser()
        args, _ = parser.parse_known_args(
            ["da", "--filter", "Interaction ID=ABC", "--filter", "Channel=Web"]
        )
        assert args.filter == ["Interaction ID=ABC", "Channel=Web"]

    def test_filter_default_is_none(self):
        """No --filter args gives None."""
        from pdstools.cli import build_parser

        parser = build_parser()
        args, _ = parser.parse_known_args(["da"])
        assert args.filter is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_da_utils.py::TestFilterCLIArgs -v`
Expected: FAIL (`args` has no attribute `filter`)

**Step 3: Implement the CLI changes**

In `python/pdstools/cli.py`, add after the `--sample` argument (after line 89):

```python
    parser.add_argument(
        "--filter",
        dest="filter",
        action="append",
        default=None,
        help=(
            "Pre-ingestion row filter for extracting specific data from large files. "
            "Specify as 'Column Name=value1,value2,...' using display names "
            "(e.g. 'Interaction ID', 'Subject ID', 'Channel'). "
            "Multiple --filter flags are ANDed together. "
            "Mutually exclusive with --sample. "
            "Exposed to the app as the PDSTOOLS_FILTER env var."
        ),
    )
```

In the env var propagation section (~line 236), add after the `args.sample` block and before `args.temp_dir`:

```python
    if args.filter:
        if args.sample:
            parser.error("--filter and --sample are mutually exclusive")
        import json
        os.environ["PDSTOOLS_FILTER"] = json.dumps(args.filter)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_da_utils.py::TestFilterCLIArgs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/pdstools/cli.py python/tests/test_da_utils.py
git commit -m "feat(filter): add --filter CLI argument with mutual exclusivity check"
```

---

### Task 4: `get_filter_specs()` helper in `streamlit_utils.py`

**Files:**
- Modify: `python/pdstools/utils/streamlit_utils.py:154` (add after `get_sample_limit`)
- Test: `python/tests/test_da_utils.py`

**Step 1: Write the failing test**

```python
class TestGetFilterSpecs:
    """Tests for get_filter_specs() env var reader."""

    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("PDSTOOLS_FILTER", raising=False)
        from pdstools.utils.streamlit_utils import get_filter_specs
        assert get_filter_specs() is None

    def test_returns_parsed_list(self, monkeypatch):
        monkeypatch.setenv("PDSTOOLS_FILTER", '["Interaction ID=ABC", "Channel=Web"]')
        from pdstools.utils.streamlit_utils import get_filter_specs
        assert get_filter_specs() == ["Interaction ID=ABC", "Channel=Web"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_da_utils.py::TestGetFilterSpecs -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `python/pdstools/utils/streamlit_utils.py` after `get_sample_limit()` (~line 155):

```python
def get_filter_specs() -> list[str] | None:
    """Return the filter specs set via ``--filter`` CLI flags.

    Returns ``None`` when no filters were configured.
    """
    raw = os.environ.get("PDSTOOLS_FILTER")
    if raw is None:
        return None
    import json
    return json.loads(raw)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_da_utils.py::TestGetFilterSpecs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/pdstools/utils/streamlit_utils.py python/tests/test_da_utils.py
git commit -m "feat(filter): add get_filter_specs() env var reader"
```

---

### Task 5: Integrate filter into Home.py

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/Home.py:130-200` (add filter branch alongside sample)

**Step 1: Implement the Home.py integration**

This is Streamlit UI code that is difficult to unit test. We integrate and
manually verify.

In `Home.py`, add `get_filter_specs` to the imports from `streamlit_utils` (line 18-25):

```python
from pdstools.utils.streamlit_utils import (
    get_data_path,
    get_filter_specs,
    get_sample_limit,
    get_temp_dir,
    show_sidebar_branding,
    show_version_header,
    standard_page_config,
)
```

Add `parse_filter_specs` to the imports from `decision_analyzer.utils` (line 12-17):

```python
from pdstools.decision_analyzer.utils import (
    parse_filter_specs,
    parse_sample_flag,
    prepare_and_save,
    should_cache_source,
    _read_source_metadata,
)
```

After line 131 (`sample_limit_raw = ...`), add filter reading:

```python
filter_specs_raw = get_filter_specs() if data_source_path else None
```

Then, inside the `if raw_data is not None:` block (line 133), add a new
`elif` branch after the sampling block (after line 188) and before the
caching `elif` (line 190):

```python
    elif filter_specs_raw:
        # Filter mode
        try:
            filter_expr = parse_filter_specs(
                filter_specs_raw,
                available_columns=set(raw_data.collect_schema().names()),
            )
        except ValueError as e:
            st.error(f"Invalid --filter value: {e}")
            st.stop()

        filter_desc = " AND ".join(filter_specs_raw)
        with st.spinner(f"Filtering data: {filter_desc}"):
            filtered = raw_data.filter(filter_expr)

            # Check for empty result before writing
            row_count = filtered.select(pl.len()).collect().item()
            if row_count == 0:
                st.warning(
                    f"Filter matched 0 rows: {filter_desc}. "
                    "Check your filter values."
                )
                st.stop()

            raw_data, sample_path = prepare_and_save(
                filtered,
                output_dir=get_temp_dir(),
                source_path=data_source_path,
            )

        st.info(
            f"🔍 Pre-ingestion filter applied: **{filter_desc}**. "
            f"**{row_count:,}** rows matched."
            + (f" Cached to `{sample_path}`." if sample_path else "")
        )
```

**Step 2: Verify no import errors**

Run: `uv run python -c "from pdstools.app.decision_analyzer.Home import *"` -- this
will fail because Streamlit isn't running, but it should not show ImportError.
Alternatively, just run the existing test suite to make sure nothing broke:

Run: `uv run pytest python/tests/test_da_utils.py -v --tb=short`
Expected: all existing tests still PASS

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/Home.py
git commit -m "feat(filter): integrate --filter into Home.py data loading pipeline"
```

---

### Task 6: End-to-end manual verification

**Step 1: Run with a known dataset and filter**

Use the minimal test CSV that exists in the repo:

```bash
uv run pdstools da --data-path data/da/sample_eev2_minimal.csv --filter "Interaction ID=INT-001"
```

Verify in the browser:
- Info banner shows the filter description
- Data loaded only contains rows for `INT-001`
- Single Decision page works with the filtered data

**Step 2: Test error cases**

```bash
# Mutual exclusivity
uv run pdstools da --data-path data/da/sample_eev2_minimal.csv --filter "Interaction ID=INT-001" --sample 100
# Expected: error message about mutual exclusivity

# Unknown column
uv run pdstools da --data-path data/da/sample_eev2_minimal.csv --filter "Bogus=X"
# Expected: error listing available columns

# No matching rows
uv run pdstools da --data-path data/da/sample_eev2_minimal.csv --filter "Interaction ID=NONEXISTENT"
# Expected: warning about 0 rows
```

**Step 3: Commit any fixes**

If any issues found, fix and commit.

---

### Task 7: Update CLI documentation

**Files:**
- Modify: `python/docs/source/GettingStartedWithDecisionAnalyzer.rst` (add `--filter` usage examples)

**Step 1: Check existing `--sample` documentation and add `--filter` in the same style**

Find where `--sample` is documented and add a parallel section for `--filter`.
Include:
- Syntax explanation
- 2-3 examples matching the ones from the design doc
- Note about mutual exclusivity with `--sample`

**Step 2: Commit**

```bash
git add python/docs/source/GettingStartedWithDecisionAnalyzer.rst
git commit -m "docs: add --filter CLI flag to Decision Analysis getting started guide"
```

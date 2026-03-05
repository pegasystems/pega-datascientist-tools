# Decision Analyzer Data Caching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add caching for non-sampled data to improve reload performance when loading 100% of data from slow formats.

**Architecture:** Extend existing `sample_and_save()` function to handle both sampling and caching modes. Rename to `prepare_and_save()` for clarity. When no sampling parameters are provided, create `decision_analyzer_cache_{count}.parquet` with 100% metadata. Add helper to detect when caching should occur.

**Tech Stack:** Polars (LazyFrame/DataFrame), Pathlib, Python 3.12+

---

## Task 1: Add helper function for cache detection

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py:802` (after `_read_source_metadata`)
- Test: `python/tests/test_da_utils.py:797` (at end)

**Step 1: Write the failing test**

Add to end of `python/tests/test_da_utils.py`:

```python
# ---------------------------------------------------------------------------
# should_cache_source
# ---------------------------------------------------------------------------


class TestShouldCacheSource:
    """Detect when source data should be cached as parquet."""

    def test_none_returns_false(self):
        from pdstools.decision_analyzer.utils import should_cache_source

        assert should_cache_source(None) is False

    def test_single_parquet_returns_false(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        parquet_file = tmp_path / "data.parquet"
        parquet_file.touch()
        assert should_cache_source(str(parquet_file)) is False

    def test_csv_file_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        csv_file = tmp_path / "data.csv"
        csv_file.touch()
        assert should_cache_source(str(csv_file)) is True

    def test_directory_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        data_dir = tmp_path / "data_dir"
        data_dir.mkdir()
        assert should_cache_source(str(data_dir)) is True

    def test_json_file_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        json_file = tmp_path / "data.json"
        json_file.touch()
        assert should_cache_source(str(json_file)) is True

    def test_arrow_file_returns_true(self, tmp_path):
        from pdstools.decision_analyzer.utils import should_cache_source

        arrow_file = tmp_path / "data.arrow"
        arrow_file.touch()
        assert should_cache_source(str(arrow_file)) is True

    def test_empty_string_returns_false(self):
        from pdstools.decision_analyzer.utils import should_cache_source

        assert should_cache_source("") is False
```

**Step 2: Run test to verify it fails**

```bash
pytest python/tests/test_da_utils.py::TestShouldCacheSource -v
```

Expected: `ImportError: cannot import name 'should_cache_source'`

**Step 3: Write minimal implementation**

Add to `python/pdstools/decision_analyzer/utils.py` after `_read_source_metadata` function (around line 832):

```python
def should_cache_source(source_path: str | None) -> bool:
    """Return True if source should be cached as parquet.

    Caching is beneficial for non-parquet sources (CSV, JSON, ZIP, directories)
    but unnecessary for single parquet files which are already optimized.

    Parameters
    ----------
    source_path : str | None
        Path to the source file or directory.

    Returns
    -------
    bool
        True if source should be cached, False otherwise.

    Examples
    --------
    >>> should_cache_source("/data/export.csv")
    True
    >>> should_cache_source("/data/export.parquet")
    False
    >>> should_cache_source(None)
    False
    """
    if not source_path:
        return False
    path = Path(source_path)
    # Skip if source is already a single parquet file
    if path.is_file() and path.suffix == ".parquet":
        return False
    return True
```

**Step 4: Run test to verify it passes**

```bash
pytest python/tests/test_da_utils.py::TestShouldCacheSource -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add python/tests/test_da_utils.py python/pdstools/decision_analyzer/utils.py
git commit -m "feat(decision_analyzer): add should_cache_source helper"
```

---

## Task 2: Add tests for caching mode in prepare_and_save

**Files:**
- Modify: `python/tests/test_da_utils.py:686` (after existing `TestSampleAndSave` class)

**Step 1: Write the failing tests**

Add to `python/tests/test_da_utils.py` after the existing `TestSampleAndSave` class:

```python
class TestPrepareAndSaveCachingMode:
    """Test prepare_and_save (renamed from sample_and_save) in caching mode."""

    @pytest.fixture()
    def mock_decision_data(self):
        """Mock decision analyzer data with interactions."""
        ids = [f"int_{i}" for i in range(10) for _ in range(2)]
        return pl.LazyFrame(
            {
                "pxInteractionID": ids,
                "value": list(range(20)),
            }
        )

    def test_cache_creates_file_with_cache_prefix(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        source_file = tmp_path / "original.csv"
        result, path = prepare_and_save(
            mock_decision_data,
            source_path=str(source_file),
            output_dir=str(tmp_path),
        )

        assert path is not None
        assert path.exists()
        # Should use "cache" prefix not "sample"
        assert "decision_analyzer_cache_" in path.name
        assert path.name.endswith(".parquet")

    def test_cache_metadata_has_100_percent(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        source_file = tmp_path / "original.csv"
        result, path = prepare_and_save(
            mock_decision_data,
            source_path=str(source_file),
            output_dir=str(tmp_path),
        )

        assert path is not None
        metadata = pl.read_parquet_metadata(str(path))
        assert metadata["pdstools:source_file"] == str(source_file)
        assert float(metadata["pdstools:sample_percentage"]) == 100.0
        assert metadata["pdstools:sample_percentage_method"] == "exact"

    def test_cache_without_source_path_returns_none(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        result, path = prepare_and_save(
            mock_decision_data,
            output_dir=str(tmp_path),
        )

        # Without source_path, caching should be skipped
        assert path is None

    def test_sampling_mode_still_uses_sample_prefix(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        result, path = prepare_and_save(
            mock_decision_data,
            fraction=0.5,
            source_path=str(tmp_path / "original.csv"),
            output_dir=str(tmp_path),
        )

        assert path is not None
        # Sampling mode should still use "sample" prefix
        assert "decision_analyzer_sample_" in path.name

    def test_cache_includes_interaction_count_in_filename(self, mock_decision_data, tmp_path):
        from pdstools.decision_analyzer.utils import prepare_and_save

        result, path = prepare_and_save(
            mock_decision_data,
            source_path=str(tmp_path / "original.csv"),
            output_dir=str(tmp_path),
        )

        assert path is not None
        # Should have formatted count (10 interactions in mock data)
        assert "10" in path.name or "10." in path.name
```

**Step 2: Run tests to verify they fail**

```bash
pytest python/tests/test_da_utils.py::TestPrepareAndSaveCachingMode -v
```

Expected: `ImportError: cannot import name 'prepare_and_save'`

**Step 3: Don't implement yet - tests are written**

We'll implement in the next task after updating the function.

**Step 4: Commit tests**

```bash
git add python/tests/test_da_utils.py
git commit -m "test(decision_analyzer): add tests for caching mode"
```

---

## Task 3: Rename sample_and_save to prepare_and_save

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py:586-721`
- Modify: `python/tests/test_da_utils.py:21-34` (import)

**Step 1: Update the function name and docstring**

In `python/pdstools/decision_analyzer/utils.py`, rename the function at line 586:

```python
def prepare_and_save(
    df: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str | None = None,
    source_path: str | None = None,
) -> tuple[pl.LazyFrame, Path | None]:
    """Prepare data for analysis by sampling or caching, and persist as parquet.

    **Sampling mode** (when n or fraction provided):
    Writes ``decision_analyzer_sample_<count>.parquet`` into *output_dir*
    (defaults to the current working directory). Returns a LazyFrame scanning
    the written file plus the file path.

    **Caching mode** (when neither n nor fraction provided):
    Writes ``decision_analyzer_cache_<count>.parquet`` into *output_dir*
    with 100% sample metadata. Useful for caching non-parquet sources (CSV,
    JSON, ZIP) for faster reloading.

    The parquet file includes metadata tracking:
    - Original source file path
    - Sample percentage relative to original data (100% for caching mode)
    - Whether percentage was calculated exactly or approximated

    If sampling is requested but the data is smaller than the requested sample,
    sampling is skipped and the original LazyFrame is returned unchanged
    (no file is written).

    Parameters
    ----------
    df : pl.LazyFrame
        Raw data to process.
    n : int, optional
        Maximum number of unique interactions to keep (sampling mode).
    fraction : float, optional
        Fraction of interactions to keep 0.0–1.0 (sampling mode).
    output_dir : str, optional
        Directory for the output parquet file. Defaults to ``"."``.
    source_path : str, optional
        Path to the original source file for metadata tracking.

    Returns
    -------
    tuple[pl.LazyFrame, Path | None]
        The (possibly sampled/cached) LazyFrame and the path to the written
        parquet file, or ``None`` when no file was written.

    Examples
    --------
    Sample data and save with metadata:

    >>> df = pl.scan_parquet("large_data.parquet")
    >>> sampled, path = prepare_and_save(
    ...     df,
    ...     n=100000,
    ...     source_path="large_data.parquet"
    ... )
    >>> print(path)
    decision_analyzer_sample_100k.parquet

    Cache non-parquet data:

    >>> df = pl.scan_csv("export.csv")
    >>> cached, path = prepare_and_save(
    ...     df,
    ...     source_path="export.csv"
    ... )
    >>> print(path)
    decision_analyzer_cache_87k.parquet

    Read metadata from a prepared file:

    >>> import polars as pl
    >>> metadata = pl.read_parquet_metadata("decision_analyzer_sample_100k.parquet")
    >>> print(metadata["pdstools:source_file"])
    large_data.parquet
    >>> print(metadata["pdstools:sample_percentage"])
    10.0
    """
```

**Step 2: Update test imports**

In `python/tests/test_da_utils.py`, update the import at line 33:

```python
from pdstools.decision_analyzer.utils import (  # noqa: E402
    _cast_columns,
    _find_interaction_id_column,
    _get_interaction_id_candidates,
    area_under_curve,
    create_hierarchical_selectors,
    get_first_level_stats,
    get_scope_config,
    gini_coefficient,
    parse_sample_flag,
    prepare_and_save,  # renamed from sample_and_save
    rename_and_cast_types,
    resolve_aliases,
    sample_interactions,
)
```

**Step 3: Update test class to reference new name**

In `python/tests/test_da_utils.py`, update line 561 class name and add alias:

```python
class TestSampleAndSave:
    """Persist sampled data as parquet."""
```

Change to:

```python
class TestPrepareAndSave:
    """Persist prepared data (sampled or cached) as parquet."""
```

**Step 4: Update all test method calls from sample_and_save to prepare_and_save**

In `python/tests/test_da_utils.py`, replace all calls to `sample_and_save` with `prepare_and_save` in the test class (lines 575-703).

Use find and replace within the `TestPrepareAndSave` class only.

**Step 5: Run tests to verify they still fail appropriately**

```bash
pytest python/tests/test_da_utils.py::TestPrepareAndSave -v
pytest python/tests/test_da_utils.py::TestPrepareAndSaveCachingMode -v
```

Expected: Import errors resolved, but tests still fail because function hasn't been updated yet.

**Step 6: Commit**

```bash
git add python/pdstools/decision_analyzer/utils.py python/tests/test_da_utils.py
git commit -m "refactor(decision_analyzer): rename sample_and_save to prepare_and_save"
```

---

## Task 3: Implement caching mode in prepare_and_save

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py:586-721`

**Step 1: Update function implementation**

Modify `prepare_and_save()` function body in `python/pdstools/decision_analyzer/utils.py`. The key changes:

1. Detect mode at the start (sampling vs caching)
2. Update filename prefix logic
3. Handle caching mode (no sampling, 100% metadata)

Replace the function body (lines 586-721) with:

```python
def prepare_and_save(
    df: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str | None = None,
    source_path: str | None = None,
) -> tuple[pl.LazyFrame, Path | None]:
    """[Keep the docstring from previous step]"""

    # Determine mode: sampling or caching
    is_sampling = (n is not None) or (fraction is not None)
    is_caching = not is_sampling

    # Skip caching if no source path provided
    if is_caching and not source_path:
        return df, None

    available = set(df.collect_schema().names())
    id_column = _find_interaction_id_column(available)

    # Step 1: Read source metadata if available
    source_metadata = None
    original_source = source_path or "unknown"
    if source_path:
        source_metadata = _read_source_metadata(source_path)
        if source_metadata:
            # Inherit original source from chained sampling
            original_source = source_metadata["source_file"]

    # Step 2: Process data based on mode
    if is_sampling:
        # Sampling mode: calculate total if needed for n-based sampling
        total = None
        if n is not None:
            total = df.select(pl.n_unique(id_column).alias("n")).collect().item()  # type: ignore[union-attribute]
            if total <= n:
                logger.info(
                    "Data has %d interactions (≤ requested %d), skipping sampling.",
                    total,
                    n,
                )
                return df, None

        # Perform sampling
        sampled = sample_interactions(df, n=n, fraction=fraction, id_column=id_column)
        processed_df = sampled.collect()  # type: ignore[union-attribute]
    else:
        # Caching mode: collect all data (no sampling)
        processed_df = df.collect()

    # Step 3: Calculate sample percentage
    if is_sampling:
        if fraction is not None:
            # Fraction-based: exact percentage
            sample_percentage = fraction * 100.0
            method = "exact"
        elif total is not None and n is not None:
            # Count-based: we already have total, calculate exact
            sample_percentage = (n / total) * 100.0
            method = "exact"
        else:
            # Fallback (shouldn't happen in normal flow)
            sample_percentage = 0.0
            method = "unknown"
    else:
        # Caching mode: always 100%
        sample_percentage = 100.0
        method = "exact"

    # Step 4: Apply inheritance if sampling a sample
    if source_metadata and is_sampling:
        # Multiply percentages for chained sampling
        source_pct = source_metadata["sample_percentage"]
        assert isinstance(source_pct, float)  # Type narrowing
        sample_percentage = (sample_percentage * source_pct) / 100.0
        # Inherit method (use "approximated" if either step was approximate)
        if source_metadata["method"] == "approximated":
            method = "approximated"

    # Step 5: Determine output filename with count
    processed_count = processed_df.select(pl.n_unique(id_column)).item()  # type: ignore[union-attr]
    formatted_count = format_count_for_filename(processed_count)

    # Choose prefix based on mode
    prefix = "decision_analyzer_sample_" if is_sampling else "decision_analyzer_cache_"

    dest = Path(output_dir) if output_dir else Path(".")
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{prefix}{formatted_count}.parquet"

    # Step 6: Build metadata
    metadata = {
        "pdstools:source_file": original_source,
        "pdstools:sample_percentage": f"{sample_percentage:.2f}",
        "pdstools:sample_percentage_method": method,
    }

    # Step 7: Write with metadata
    logger.info("Writing prepared data to %s", out_path)
    processed_df.write_parquet(out_path, metadata=metadata)

    return pl.scan_parquet(out_path), out_path
```

**Step 2: Run tests to verify they pass**

```bash
pytest python/tests/test_da_utils.py::TestPrepareAndSave -v
pytest python/tests/test_da_utils.py::TestPrepareAndSaveCachingMode -v
```

Expected: All tests PASS

**Step 3: Commit**

```bash
git add python/pdstools/decision_analyzer/utils.py
git commit -m "feat(decision_analyzer): implement caching mode in prepare_and_save"
```

---

## Task 4: Update imports in Home.py

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/Home.py:12`

**Step 1: Update import statement**

Change line 12 from:

```python
from pdstools.decision_analyzer.utils import parse_sample_flag, sample_and_save, _read_source_metadata
```

To:

```python
from pdstools.decision_analyzer.utils import parse_sample_flag, prepare_and_save, should_cache_source, _read_source_metadata
```

**Step 2: Verify no syntax errors**

```bash
python -c "from pdstools.app.decision_analyzer.Home import *"
```

Expected: No import errors

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/Home.py
git commit -m "refactor(decision_analyzer): update imports for prepare_and_save"
```

---

## Task 5: Add caching logic to Home.py

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/Home.py:92-125`

**Step 1: Refactor data preparation section**

Replace lines 92-125 with unified sampling/caching logic:

```python
# Pre-ingestion data preparation (sampling or caching)
sample_limit_raw = get_sample_limit()
if raw_data is not None:
    prepared_path = None  # Track the prepared file path

    if sample_limit_raw:
        # Sampling mode
        try:
            sample_kwargs = parse_sample_flag(sample_limit_raw)
        except ValueError as e:
            st.error(f"Invalid --sample value: {e}")
            st.stop()

        # Build explicit sampling message based on parameters
        if "fraction" in sample_kwargs:
            sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the interactions…"
        elif "n" in sample_kwargs:
            sampling_msg = f"Sampling {sample_kwargs['n']:,} interactions…"
        else:
            sampling_msg = "Sampling interactions…"

        with st.spinner(sampling_msg):
            raw_data, prepared_path = prepare_and_save(
                raw_data,
                n=sample_kwargs.get("n"),  # type: ignore[arg-type]
                fraction=sample_kwargs.get("fraction"),  # type: ignore[arg-type]
                output_dir=get_temp_dir(),
                source_path=data_source_path,
            )

        label = sample_limit_raw.strip()
        if prepared_path is not None:
            st.info(
                f"📉 Pre-ingestion sampling applied: keeping **{label}** interactions. "
                f"Sampled data saved to `{prepared_path}`."
            )
        else:
            st.info(f"📉 Sampling requested (**{label}**) but data already within limit — using full dataset.")

    elif should_cache_source(data_source_path):
        # Caching mode - save 100% of data from non-parquet sources
        with st.spinner("Caching data for faster reloading..."):
            raw_data, prepared_path = prepare_and_save(
                raw_data,
                source_path=data_source_path,
                output_dir=".",  # Current working directory
            )

        if prepared_path is not None:
            st.info(f"💾 Cached data saved to `{prepared_path}` for faster reloading.")
```

**Step 2: Test in development**

Start the app and test both modes:

```bash
# Test sampling mode
pdstools da --sample 50000 --data-path test_data.csv

# Test caching mode (no --sample flag)
pdstools da --data-path test_data.csv
```

Expected:
- Sampling mode: creates `decision_analyzer_sample_*` file in temp dir
- Caching mode: creates `decision_analyzer_cache_*` file in current dir
- Both show appropriate messages

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/Home.py
git commit -m "feat(decision_analyzer): add caching for non-sampled data"
```

---

## Task 6: Update all other references to sample_and_save

**Files:**
- Check: All files found by grep earlier

**Step 1: Search for remaining references**

```bash
grep -r "sample_and_save" --include="*.py" python/ docs/
```

**Step 2: Update remaining imports and calls**

For each file found:
- Update imports: `sample_and_save` → `prepare_and_save`
- Update function calls (should be no behavior changes)
- Update any docstring references

Expected files to update:
- `python/pdstools/decision_analyzer/SPECS.md` (documentation only)
- Any other Python files that import the function

**Step 3: Verify all references updated**

```bash
grep -r "sample_and_save" --include="*.py" python/
```

Expected: Only matches in comments or strings, no actual code references

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(decision_analyzer): complete rename of sample_and_save to prepare_and_save"
```

---

## Task 7: Run full test suite

**Files:**
- Test: All decision analyzer tests

**Step 1: Run all DA utils tests**

```bash
pytest python/tests/test_da_utils.py -v
```

Expected: All tests PASS

**Step 2: Run broader test suite**

```bash
pytest python/tests/ -k decision -v
```

Expected: All decision analyzer tests PASS

**Step 3: Fix any failures**

If tests fail, investigate and fix:
- Missing imports
- Incorrect function calls
- Test expectations needing updates

**Step 4: Final commit if fixes needed**

```bash
git add -A
git commit -m "fix(tests): fix decision analyzer test issues"
```

---

## Task 8: Update documentation

**Files:**
- Modify: `python/pdstools/decision_analyzer/SPECS.md:39`

**Step 1: Update SPECS.md**

In `python/pdstools/decision_analyzer/SPECS.md`, update line 39 from:

```markdown
- [x] **Sample metadata tracking and display** — Implemented comprehensive metadata tracking for sampled files. The `sample_and_save` function now: ...
```

To:

```markdown
- [x] **Sample metadata tracking and display** — Implemented comprehensive metadata tracking for sampled files. The `prepare_and_save` function now: (1) Generates descriptive filenames with human-readable counts (e.g., `decision_analyzer_sample_87k.parquet` or `decision_analyzer_cache_49k.parquet`), (2) Stores metadata in parquet files including original source file path, sample percentage, and calculation method (exact/approximate), (3) Supports chained sampling with percentage multiplication and lineage tracking, (4) Automatically caches non-sampled data from slow formats (CSV, JSON, ZIP) as `decision_analyzer_cache_*` files with 100% metadata for faster reloading. The Home and Overview pages display sample information when data is sampled. Uses Polars native metadata API with keys: `pdstools:source_file`, `pdstools:sample_percentage`, `pdstools:sample_percentage_method`. Reading: `metadata = pl.read_parquet_metadata(file_path)`. The sampling UI message now shows explicit parameters (e.g., "Sampling 40% of the interactions…" or "Sampling 100,000 interactions…").
```

**Step 2: Commit**

```bash
git add python/pdstools/decision_analyzer/SPECS.md
git commit -m "docs(decision_analyzer): update SPECS.md with caching feature"
```

---

## Task 9: Manual verification

**Files:**
- Test: Real-world data loading scenarios

**Step 1: Test with CSV file**

```bash
# Create or use an existing CSV file with decision data
pdstools da --data-path examples/test_data.csv
```

Expected:
- Message: "💾 Cached data saved to `decision_analyzer_cache_*` for faster reloading."
- File created in current directory
- App loads successfully

**Step 2: Test with parquet file (should skip caching)**

```bash
pdstools da --data-path examples/test_data.parquet
```

Expected:
- NO caching message (parquet already optimized)
- NO cache file created
- App loads successfully

**Step 3: Test with sampling (should use sample prefix)**

```bash
pdstools da --data-path examples/test_data.csv --sample 50000
```

Expected:
- Message: "📉 Pre-ingestion sampling applied..."
- File created in temp dir with `sample_` prefix
- App loads successfully

**Step 4: Verify metadata**

```python
import polars as pl
metadata = pl.read_parquet_metadata("decision_analyzer_cache_49k.parquet")
print(metadata["pdstools:source_file"])
print(metadata["pdstools:sample_percentage"])  # Should be "100.00"
print(metadata["pdstools:sample_percentage_method"])  # Should be "exact"
```

**Step 5: Document any issues found**

If issues arise, create follow-up commits to fix them.

---

## Completion Checklist

- [ ] Helper function `should_cache_source()` implemented and tested
- [ ] Function renamed from `sample_and_save` to `prepare_and_save`
- [ ] Caching mode implemented (creates `decision_analyzer_cache_*` files)
- [ ] Metadata correctly set with 100% for cached files
- [ ] Home.py updated to call caching logic
- [ ] All imports updated throughout codebase
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Manual verification with real data complete

---

## Notes for Engineer

**Key implementation points:**
- The function signature doesn't change - backward compatible
- Mode detection: `is_sampling = (n is not None) or (fraction is not None)`
- Filename prefix: `"decision_analyzer_sample_"` or `"decision_analyzer_cache_"`
- Caching skipped if `source_path` is None (can't track metadata without it)
- Storage location differs: sampling uses temp dir, caching uses current dir

**Testing strategy:**
- Unit tests cover both modes independently
- Existing sampling tests ensure no regression
- New caching tests verify 100% metadata and cache prefix
- Manual verification ensures real-world usability

**DRY principles:**
- Reuse existing `format_count_for_filename()`
- Reuse existing metadata structure
- Single unified function handles both operations
- Minimal code duplication

**YAGNI principles:**
- No auto-reuse of cached files (user said "always rebuild")
- No cache invalidation logic (keep it simple)
- No compression settings optimization (use defaults)
- No progress callbacks or advanced features

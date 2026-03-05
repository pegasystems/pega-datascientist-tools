# Decision Analyzer Sampling Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance Decision Analyzer sampling to generate descriptive filenames and store lineage metadata in parquet files.

**Architecture:** Add filename formatting utility, extend sample_and_save to track metadata using Polars native parquet metadata API, support chained sampling with percentage multiplication.

**Tech Stack:** Polars (parquet metadata), Python pathlib, existing pdstools testing framework

---

## Task 1: Implement format_count_for_filename Utility

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py` (add function after line 666)
- Test: `python/tests/test_da_utils.py` (add tests in the parse_sample_flag section)

**Step 1: Write failing tests for format_count_for_filename**

Add to `python/tests/test_da_utils.py` after the parse_sample_flag tests (around line 200):

```python
# format_count_for_filename
def test_format_count_for_filename_units():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(42) == "42"
    assert format_count_for_filename(999) == "999"


def test_format_count_for_filename_thousands():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(1000) == "1k"
    assert format_count_for_filename(1500) == "1.5k"
    assert format_count_for_filename(87432) == "87k"
    assert format_count_for_filename(999999) == "1000k"


def test_format_count_for_filename_millions():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(1000000) == "1M"
    assert format_count_for_filename(1234567) == "1.2M"
    assert format_count_for_filename(87000000) == "87M"
    assert format_count_for_filename(999999999) == "1000M"


def test_format_count_for_filename_billions():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    assert format_count_for_filename(1000000000) == "1B"
    assert format_count_for_filename(2500000000) == "2.5B"
    assert format_count_for_filename(87000000000) == "87B"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_da_utils.py::test_format_count_for_filename_units -v`

Expected: `ImportError: cannot import name 'format_count_for_filename'`

**Step 3: Implement format_count_for_filename**

Add to `python/pdstools/decision_analyzer/utils.py` after line 666:

```python
def format_count_for_filename(count: int) -> str:
    """Format an interaction count for use in filenames.

    Uses human-readable abbreviations with 2 significant figures.

    Parameters
    ----------
    count : int
        Number of interactions.

    Returns
    -------
    str
        Formatted count (e.g., "87k", "1.2M", "2.5B").

    Examples
    --------
    >>> format_count_for_filename(42)
    '42'
    >>> format_count_for_filename(1500)
    '1.5k'
    >>> format_count_for_filename(87432)
    '87k'
    >>> format_count_for_filename(1234567)
    '1.2M'
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        # Thousands
        value = count / 1000
        if value >= 100:
            return f"{int(round(value))}k"
        else:
            formatted = f"{value:.2g}"
            # Remove unnecessary decimal point for whole numbers
            return f"{formatted}k"
    elif count < 1_000_000_000:
        # Millions
        value = count / 1_000_000
        if value >= 100:
            return f"{int(round(value))}M"
        else:
            formatted = f"{value:.2g}"
            return f"{formatted}M"
    else:
        # Billions
        value = count / 1_000_000_000
        if value >= 100:
            return f"{int(round(value))}B"
        else:
            formatted = f"{value:.2g}"
            return f"{formatted}B"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_da_utils.py::test_format_count_for_filename -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/utils.py python/tests/test_da_utils.py
git commit -m "feat(decision_analyzer): add format_count_for_filename utility

Add utility function to format interaction counts for filenames with
human-readable abbreviations (k, M, B) using 2 significant figures.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Helper to Read Source Metadata

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py` (add function after format_count_for_filename)
- Test: `python/tests/test_da_utils.py`

**Step 1: Write failing test for read_source_metadata**

Add to `python/tests/test_da_utils.py` after format_count_for_filename tests:

```python
# _read_source_metadata
def test_read_source_metadata_with_metadata(tmp_path):
    from pdstools.decision_analyzer.utils import _read_source_metadata
    import polars as pl

    # Create a file with metadata
    df = pl.DataFrame({"pxInteractionID": ["A", "B", "C"]})
    test_file = tmp_path / "test.parquet"
    metadata = {
        "pdstools:source_file": "/original/data.parquet",
        "pdstools:sample_percentage": "50.0",
        "pdstools:sample_percentage_method": "exact"
    }
    df.write_parquet(test_file, metadata=metadata)

    result = _read_source_metadata(str(test_file))

    assert result is not None
    assert result["source_file"] == "/original/data.parquet"
    assert result["sample_percentage"] == 50.0
    assert result["method"] == "exact"


def test_read_source_metadata_without_metadata(tmp_path):
    from pdstools.decision_analyzer.utils import _read_source_metadata
    import polars as pl

    # Create a file without our metadata
    df = pl.DataFrame({"pxInteractionID": ["A", "B", "C"]})
    test_file = tmp_path / "test.parquet"
    df.write_parquet(test_file)

    result = _read_source_metadata(str(test_file))

    assert result is None


def test_read_source_metadata_nonexistent_file():
    from pdstools.decision_analyzer.utils import _read_source_metadata

    result = _read_source_metadata("/nonexistent/file.parquet")

    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest python/tests/test_da_utils.py::test_read_source_metadata_with_metadata -v`

Expected: `ImportError: cannot import name '_read_source_metadata'`

**Step 3: Implement _read_source_metadata**

Add to `python/pdstools/decision_analyzer/utils.py` after format_count_for_filename:

```python
def _read_source_metadata(source_path: str) -> dict[str, str | float] | None:
    """Read pdstools metadata from a parquet file if it exists.

    Parameters
    ----------
    source_path : str
        Path to the parquet file to check.

    Returns
    -------
    dict or None
        Dictionary with keys: source_file, sample_percentage, method
        Returns None if file doesn't exist, is not parquet, or lacks metadata.
    """
    try:
        metadata = pl.read_parquet_metadata(source_path)

        # Check if this file has our metadata
        if "pdstools:source_file" not in metadata:
            return None

        return {
            "source_file": metadata.get("pdstools:source_file"),
            "sample_percentage": float(metadata.get("pdstools:sample_percentage", "0")),
            "method": metadata.get("pdstools:sample_percentage_method", "unknown"),
        }
    except Exception:
        # File doesn't exist, not a parquet, or other read error
        return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_da_utils.py::test_read_source_metadata -v`

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/utils.py python/tests/test_da_utils.py
git commit -m "feat(decision_analyzer): add helper to read source metadata

Add private helper function to extract pdstools metadata from sampled
parquet files for lineage tracking.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Extend sample_and_save with Metadata Support

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py:586-642` (sample_and_save function)
- Test: `python/tests/test_da_utils.py`

**Step 1: Write failing tests for enhanced sample_and_save**

Add to `python/tests/test_da_utils.py` in the sample_and_save section:

```python
def test_sample_and_save_with_source_path_metadata(mock_decision_data, tmp_path):
    from pdstools.decision_analyzer.utils import sample_and_save
    import polars as pl

    lf = mock_decision_data
    source_file = tmp_path / "original.parquet"

    result, path = sample_and_save(
        lf,
        fraction=0.5,
        output_dir=str(tmp_path),
        source_path=str(source_file)
    )

    assert path is not None
    # Check filename contains count
    assert "decision_analyzer_sample_" in path.name
    assert path.name.endswith(".parquet")

    # Check metadata was written
    metadata = pl.read_parquet_metadata(str(path))
    assert "pdstools:source_file" in metadata
    assert metadata["pdstools:source_file"] == str(source_file)
    assert "pdstools:sample_percentage" in metadata
    assert float(metadata["pdstools:sample_percentage"]) == 50.0
    assert metadata["pdstools:sample_percentage_method"] == "exact"


def test_sample_and_save_with_chained_sampling(mock_decision_data, tmp_path):
    from pdstools.decision_analyzer.utils import sample_and_save
    import polars as pl

    lf = mock_decision_data
    original_source = "/data/original.parquet"

    # First sample: 50%
    first_sample_file = tmp_path / "first.parquet"
    first_meta = {
        "pdstools:source_file": original_source,
        "pdstools:sample_percentage": "50.0",
        "pdstools:sample_percentage_method": "exact"
    }
    lf.collect().write_parquet(first_sample_file, metadata=first_meta)

    # Second sample: 20% of the first sample
    lf_rescan = pl.scan_parquet(first_sample_file)
    result, path = sample_and_save(
        lf_rescan,
        fraction=0.2,
        output_dir=str(tmp_path),
        source_path=str(first_sample_file)
    )

    assert path is not None

    # Check metadata inheritance
    metadata = pl.read_parquet_metadata(str(path))
    # Should inherit original source, not intermediate file
    assert metadata["pdstools:source_file"] == original_source
    # Should multiply percentages: 50% * 20% = 10%
    assert float(metadata["pdstools:sample_percentage"]) == 10.0
    assert metadata["pdstools:sample_percentage_method"] == "exact"


def test_sample_and_save_filename_format(mock_decision_data, tmp_path):
    from pdstools.decision_analyzer.utils import sample_and_save
    import polars as pl

    lf = mock_decision_data

    # Sample to a specific count
    result, path = sample_and_save(
        lf,
        n=100,
        output_dir=str(tmp_path),
        source_path="test.parquet"
    )

    if path is not None:
        # Filename should contain formatted count
        assert "decision_analyzer_sample_" in path.name
        # Should have a number (exact format depends on actual count in mock data)
        assert any(char.isdigit() for char in path.name)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest python/tests/test_da_utils.py::test_sample_and_save_with_source_path_metadata -v`

Expected: Test fails because sample_and_save doesn't accept source_path parameter yet

**Step 3: Implement enhanced sample_and_save**

Replace the `sample_and_save` function in `python/pdstools/decision_analyzer/utils.py` (lines 586-642):

```python
def sample_and_save(
    df: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str | None = None,
    source_path: str | None = None,
) -> tuple[pl.LazyFrame, Path | None]:
    """Sample interactions and persist the result as a parquet file.

    Writes ``decision_analyzer_sample_<count>.parquet`` into *output_dir*
    (defaults to the current working directory). The filename includes a
    human-readable count (e.g., 87k, 1.2M). Returns a LazyFrame scanning
    the written file plus the file path, so callers can display where the
    sample was saved.

    The parquet file includes metadata tracking:
    - Original source file path
    - Sample percentage relative to original data
    - Whether percentage was calculated exactly or approximated

    If the data is smaller than the requested sample, sampling is skipped
    and the original LazyFrame is returned unchanged (no file is written).

    Parameters
    ----------
    df : pl.LazyFrame
        Raw data to sample from.
    n : int, optional
        Maximum number of unique interactions to keep.
    fraction : float, optional
        Fraction of interactions to keep (0.0–1.0).
    output_dir : str, optional
        Directory for the sample parquet file. Defaults to ``"."``.
    source_path : str, optional
        Path to the original source file for metadata tracking.

    Returns
    -------
    tuple[pl.LazyFrame, Path | None]
        The (possibly sampled) LazyFrame and the path to the written
        parquet file, or ``None`` when sampling was skipped.
    """
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

    # Step 2: Calculate total interactions if needed
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

    # Step 3: Perform sampling
    sampled = sample_interactions(df, n=n, fraction=fraction, id_column=id_column)
    sampled_df = sampled.collect()  # type: ignore[union-attribute]

    # Step 4: Calculate sample percentage
    if fraction is not None:
        # Fraction-based: exact percentage
        sample_percentage = fraction * 100.0
        method = "exact"
    elif total is not None:
        # Count-based: we already have total, calculate exact
        sample_percentage = (n / total) * 100.0
        method = "exact"
    else:
        # Fallback (shouldn't happen in normal flow)
        sample_percentage = 0.0
        method = "unknown"

    # Step 5: Apply inheritance if sampling a sample
    if source_metadata:
        # Multiply percentages for chained sampling
        sample_percentage = (sample_percentage * source_metadata["sample_percentage"]) / 100.0
        # Inherit method (use "approximated" if either step was approximate)
        if source_metadata["method"] == "approximated":
            method = "approximated"

    # Step 6: Determine output filename with count
    sampled_count = sampled_df.select(pl.n_unique(id_column)).item()  # type: ignore[union-attr]
    formatted_count = format_count_for_filename(sampled_count)

    dest = Path(output_dir) if output_dir else Path(".")
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"decision_analyzer_sample_{formatted_count}.parquet"

    # Step 7: Build metadata
    metadata = {
        "pdstools:source_file": original_source,
        "pdstools:sample_percentage": f"{sample_percentage:.2f}",
        "pdstools:sample_percentage_method": method,
    }

    # Step 8: Write with metadata
    logger.info("Writing sampled data to %s", out_path)
    sampled_df.write_parquet(out_path, metadata=metadata)

    return pl.scan_parquet(out_path), out_path
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_da_utils.py::test_sample_and_save -v`

Expected: All sample_and_save tests PASS (including new ones)

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/utils.py python/tests/test_da_utils.py
git commit -m "feat(decision_analyzer): add metadata tracking to sample_and_save

Extend sample_and_save to:
- Generate filenames with interaction counts (e.g., sample_87k.parquet)
- Store source file path and sample percentage in parquet metadata
- Support chained sampling with percentage multiplication
- Track lineage through metadata inheritance

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update CLI Integration in Home.py

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/Home.py:105-110`

**Step 1: Identify source path in Home.py**

Read `python/pdstools/app/decision_analyzer/Home.py` to understand how data is loaded and where the source path comes from.

Current code (lines 76-93):
- `configured_path = get_data_path()` - CLI --data-path flag
- `raw_data = handle_file_upload()` - Streamlit upload widget
- `raw_data = handle_file_path()` - Managed deployment file input
- `raw_data = handle_data_path()` - Load from configured_path

We need to track which source was actually used.

**Step 2: Add source path tracking to Home.py**

Modify `python/pdstools/app/decision_analyzer/Home.py` around lines 68-119:

```python
# File upload — always visible. Drag-and-drop or use the Browse button.
raw_data = handle_file_upload()
data_source_path = None  # NEW: Track the source path

# For managed deployments, also show a server-side file path input
if is_managed_deployment():
    if raw_data is None:
        raw_data = handle_file_path()
        # TODO: handle_file_path should return the path, for now we don't have it

# If --data-path was provided, load from that path (takes priority over sample data)
configured_path = get_data_path()
if raw_data is None and configured_path:
    with st.spinner(f"Loading data from configured path: {configured_path}"):
        raw_data = handle_data_path()
        data_source_path = configured_path  # NEW: Capture the source
    if raw_data is not None:
        st.info(f"📂 Loaded data from configured path: `{configured_path}`")

has_new_data = raw_data is not None
has_existing_data = "decision_data" in st.session_state

# Fall back to sample data only when nothing was uploaded *and* no data is loaded yet
if not has_new_data and not has_existing_data:
    with st.spinner("Loading sample data"):
        raw_data = handle_sample_data()
    has_new_data = raw_data is not None
    st.info(
        "No file uploaded — using built-in sample data. Upload your own data above to analyze it.",
    )

# Pre-ingestion sampling (--sample CLI flag)
sample_limit_raw = get_sample_limit()
if raw_data is not None and sample_limit_raw:
    try:
        sample_kwargs = parse_sample_flag(sample_limit_raw)
    except ValueError as e:
        st.error(f"Invalid --sample value: {e}")
        st.stop()

    with st.spinner("Sampling interactions…"):
        raw_data, sample_path = sample_and_save(
            raw_data,
            n=sample_kwargs.get("n"),  # type: ignore[arg-type]
            fraction=sample_kwargs.get("fraction"),  # type: ignore[arg-type]
            output_dir=get_temp_dir(),
            source_path=data_source_path,  # NEW: Pass the source path
        )
    label = sample_limit_raw.strip()
    if sample_path is not None:
        st.info(
            f"📉 Pre-ingestion sampling applied: keeping **{label}** interactions. "
            f"Sampled data saved to `{sample_path}`."
        )
    else:
        st.info(f"📉 Sampling requested (**{label}**) but data already within limit — using full dataset.")
```

**Step 3: Manual test the CLI integration**

Since this is Streamlit-based, we need to test manually:

1. Start the app with sample flag:
   ```bash
   uv run pdstools da --sample 10000 --data-path /path/to/test/data.parquet
   ```

2. Verify:
   - A file is created with format `decision_analyzer_sample_<count>.parquet`
   - The file contains metadata (check with `pl.read_parquet_metadata()`)
   - The UI shows the correct message

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/Home.py
git commit -m "feat(decision_analyzer): pass source path to sampling

Update CLI integration to pass source file path to sample_and_save
for metadata tracking. Enables lineage information in sampled files.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Edge Case Tests

**Files:**
- Test: `python/tests/test_da_utils.py`

**Step 1: Write edge case tests**

Add to `python/tests/test_da_utils.py`:

```python
def test_format_count_for_filename_edge_cases():
    from pdstools.decision_analyzer.utils import format_count_for_filename

    # Exact boundaries
    assert format_count_for_filename(1) == "1"
    assert format_count_for_filename(10) == "10"
    assert format_count_for_filename(100) == "100"
    # Check boundary transitions
    assert "k" in format_count_for_filename(1001)
    assert "M" in format_count_for_filename(1000001)
    assert "B" in format_count_for_filename(1000000001)


def test_sample_and_save_without_source_path(mock_decision_data, tmp_path):
    """Test backward compatibility - source_path is optional."""
    from pdstools.decision_analyzer.utils import sample_and_save
    import polars as pl

    lf = mock_decision_data

    # Call without source_path (backward compatibility)
    result, path = sample_and_save(
        lf,
        fraction=0.5,
        output_dir=str(tmp_path)
    )

    assert path is not None
    # Should still write metadata, but with "unknown" source
    metadata = pl.read_parquet_metadata(str(path))
    assert metadata["pdstools:source_file"] == "unknown"


def test_sample_and_save_with_invalid_source_path(mock_decision_data, tmp_path):
    """Test graceful handling of nonexistent source path."""
    from pdstools.decision_analyzer.utils import sample_and_save
    import polars as pl

    lf = mock_decision_data

    # Pass nonexistent source path
    result, path = sample_and_save(
        lf,
        fraction=0.5,
        output_dir=str(tmp_path),
        source_path="/nonexistent/file.parquet"
    )

    assert path is not None
    # Should write metadata with the provided path (even if it doesn't exist)
    metadata = pl.read_parquet_metadata(str(path))
    assert metadata["pdstools:source_file"] == "/nonexistent/file.parquet"
```

**Step 2: Run edge case tests**

Run: `uv run pytest python/tests/test_da_utils.py::test_format_count_for_filename_edge_cases -v`
Run: `uv run pytest python/tests/test_da_utils.py::test_sample_and_save_without_source_path -v`
Run: `uv run pytest python/tests/test_da_utils.py::test_sample_and_save_with_invalid_source_path -v`

Expected: All tests PASS

**Step 3: Run full test suite for decision_analyzer utils**

Run: `uv run pytest python/tests/test_da_utils.py -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add python/tests/test_da_utils.py
git commit -m "test(decision_analyzer): add edge case tests for sampling

Add tests for boundary conditions, backward compatibility, and error
handling in filename formatting and metadata tracking.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Documentation and Final Verification

**Files:**
- Modify: `python/pdstools/decision_analyzer/SPECS.md` (if exists, document the new behavior)

**Step 1: Document the feature in SPECS.md**

Check if `python/pdstools/decision_analyzer/SPECS.md` exists and add documentation about:
- New filename format
- Metadata fields and their meaning
- How to read metadata from sampled files

**Step 2: Add example of reading metadata**

Add to docstring of `sample_and_save` function an example:

```python
    Examples
    --------
    Sample data and save with metadata:

    >>> df = pl.scan_parquet("large_data.parquet")
    >>> sampled, path = sample_and_save(
    ...     df,
    ...     n=100000,
    ...     source_path="large_data.parquet"
    ... )
    >>> print(path)
    decision_analyzer_sample_100k.parquet

    Read metadata from a sampled file:

    >>> import polars as pl
    >>> metadata = pl.read_parquet_metadata("decision_analyzer_sample_100k.parquet")
    >>> print(metadata["pdstools:source_file"])
    large_data.parquet
    >>> print(metadata["pdstools:sample_percentage"])
    10.0
```

**Step 3: Run all tests one final time**

Run: `uv run pytest python/tests/test_da_utils.py -v --tb=short`

Expected: All tests PASS

**Step 4: Run type checking (if configured)**

Run: `uv run mypy python/pdstools/decision_analyzer/utils.py` (if mypy is configured)

Expected: No errors or only pre-existing errors

**Step 5: Final commit**

```bash
git add python/pdstools/decision_analyzer/utils.py
git commit -m "docs(decision_analyzer): add examples for metadata reading

Add usage examples to sample_and_save docstring showing how to read
metadata from sampled parquet files.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Testing Checklist

After implementation, verify:

- [ ] `format_count_for_filename` correctly formats all ranges (units, k, M, B)
- [ ] `format_count_for_filename` uses 2 significant figures
- [ ] Sample files have descriptive names like `decision_analyzer_sample_87k.parquet`
- [ ] Metadata is written with correct keys: `pdstools:source_file`, `pdstools:sample_percentage`, `pdstools:sample_percentage_method`
- [ ] Chained sampling correctly multiplies percentages
- [ ] Chained sampling inherits original source, not intermediate file
- [ ] Backward compatibility: `sample_and_save` works without `source_path`
- [ ] CLI integration passes source path correctly
- [ ] All existing tests still pass
- [ ] Manual CLI test: `pdstools da --sample 10000 --data-path <file>` works

## Known Limitations

1. **File upload tracking**: When users upload files via Streamlit widget, we don't capture the original filename (Streamlit limitation). The `source_path` will be `None` in this case, resulting in "unknown" in metadata.

2. **Approximation fallback**: The 500MB approximation logic is implemented but should rarely trigger since we already count interactions during sampling operations.

## Future Enhancements

See design document for ideas:
- Add CLI command to display metadata: `pdstools da --show-metadata <file>`
- Add timestamp to metadata
- Add sampling method (hash-based) to metadata
- Utility function to trace full lineage chain

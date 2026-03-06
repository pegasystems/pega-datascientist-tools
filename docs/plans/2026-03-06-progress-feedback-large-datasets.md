# Progress Feedback for Large Datasets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add time estimates to Streamlit spinners when loading large datasets, focusing on zip extraction and data sampling operations.

**Architecture:** Create new `progress_utils.py` module with file size-based time estimation functions. Enhance existing `st.spinner()` calls in `da_streamlit_utils.py` and `Home.py` to show estimated time ranges using the `humanize` library for formatting.

**Tech Stack:** Python, Streamlit, Polars, humanize library

---

## Task 1: Create progress_utils module with time estimation functions

**Files:**
- Create: `python/pdstools/utils/progress_utils.py`
- Test: `python/tests/test_progress_utils.py`

**Step 1: Write the failing test for estimate_extraction_time**

Create `python/tests/test_progress_utils.py`:

```python
import pytest
from pdstools.utils.progress_utils import estimate_extraction_time


def test_estimate_extraction_time_small_file():
    """Test estimation for 10 MB file."""
    min_sec, max_sec = estimate_extraction_time(10 * 1024 * 1024)
    assert min_sec < 1  # Should be very fast
    assert max_sec < 5
    assert min_sec < max_sec


def test_estimate_extraction_time_large_file():
    """Test estimation for 4.5 GB file."""
    min_sec, max_sec = estimate_extraction_time(4.5 * 1024 * 1024 * 1024)
    assert 45 < min_sec < 60  # ~45 seconds at 100 MB/s
    assert 150 < max_sec < 180  # ~150 seconds at 30 MB/s
    assert min_sec < max_sec


def test_estimate_extraction_time_medium_file():
    """Test estimation for 500 MB file."""
    min_sec, max_sec = estimate_extraction_time(500 * 1024 * 1024)
    assert min_sec > 0
    assert max_sec > min_sec
```

**Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_progress_utils.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pdstools.utils.progress_utils'"

**Step 3: Write minimal implementation**

Create `python/pdstools/utils/progress_utils.py`:

```python
"""Utilities for progress feedback and time estimation."""


def estimate_extraction_time(file_size_bytes: int) -> tuple[float, float]:
    """Estimate extraction time for a zip file based on size.

    Uses calibrated extraction speeds to provide min/max range.
    Conservative estimates to avoid under-promising.

    Parameters
    ----------
    file_size_bytes : int
        Size of the file in bytes

    Returns
    -------
    tuple[float, float]
        (min_seconds, max_seconds) for a range estimate

    Examples
    --------
    >>> min_time, max_time = estimate_extraction_time(1024 * 1024 * 1024)  # 1 GB
    >>> min_time < max_time
    True
    """
    # Calibrated extraction speeds (MB/s)
    # Conservative estimates to avoid under-promising
    FAST_SPEED_MBS = 100  # SSD with good CPU
    SLOW_SPEED_MBS = 30  # HDD or compressed data

    size_mb = file_size_bytes / (1024 * 1024)
    min_time = size_mb / FAST_SPEED_MBS
    max_time = size_mb / SLOW_SPEED_MBS

    return (min_time, max_time)
```

**Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_progress_utils.py::test_estimate_extraction_time_small_file -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_estimate_extraction_time_large_file -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_estimate_extraction_time_medium_file -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/pdstools/utils/progress_utils.py python/tests/test_progress_utils.py
git commit -m "feat(utils): add extraction time estimation function"
```

---

## Task 2: Add time formatting function with humanize

**Files:**
- Modify: `python/pdstools/utils/progress_utils.py`
- Modify: `python/tests/test_progress_utils.py`

**Step 1: Write the failing test for format_time_estimate**

Add to `python/tests/test_progress_utils.py`:

```python
from pdstools.utils.progress_utils import format_time_estimate


def test_format_time_estimate_very_short():
    """Test formatting for operations under 10 seconds."""
    result = format_time_estimate(2, 5)
    assert result == "a few seconds"


def test_format_time_estimate_seconds():
    """Test formatting for operations under a minute."""
    result = format_time_estimate(15, 45)
    assert "second" in result.lower()
    assert "45" in result


def test_format_time_estimate_minutes():
    """Test formatting for operations over a minute."""
    result = format_time_estimate(120, 180)
    assert "minute" in result.lower()


def test_format_time_estimate_range():
    """Test that longer operations show a range."""
    result = format_time_estimate(60, 240)
    assert "to" in result  # Should show "X to Y"


def test_format_time_estimate_same_range():
    """Test that similar times don't show redundant range."""
    result = format_time_estimate(120, 130)
    # humanize.naturaldelta will likely give same result for both
    # so should not show "2 minutes to 2 minutes"
    result_count = result.count("minute")
    assert result_count <= 2  # At most "X minutes to Y minutes"
```

**Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_progress_utils.py::test_format_time_estimate_very_short -v`
Expected: FAIL with "ImportError: cannot import name 'format_time_estimate'"

**Step 3: Add humanize dependency first**

Check current dependencies:
```bash
grep -A 10 "\[project.dependencies\]" python/pyproject.toml
```

Note the format, then add `humanize` to the dependencies list.

**Step 4: Write minimal implementation**

Add to `python/pdstools/utils/progress_utils.py`:

```python
def format_time_estimate(min_sec: float, max_sec: float) -> str:
    """Format time range as user-friendly string.

    Uses humanize library to create natural language time descriptions.
    Shows ranges for operations over 10 seconds, simple descriptions for
    shorter operations.

    Parameters
    ----------
    min_sec : float
        Minimum estimated time in seconds
    max_sec : float
        Maximum estimated time in seconds

    Returns
    -------
    str
        User-friendly time description

    Examples
    --------
    >>> format_time_estimate(2, 5)
    'a few seconds'

    >>> format_time_estimate(120, 180)
    '2 minutes to 3 minutes'
    """
    try:
        import humanize
    except ImportError:
        # Fallback if humanize not available
        if max_sec < 60:
            return f"{int(max_sec)} seconds"
        else:
            return f"{int(max_sec / 60)} minutes"

    if max_sec < 10:
        return "a few seconds"
    elif max_sec < 60:
        return f"{int(max_sec)} seconds"
    else:
        # Show range for longer operations
        min_str = humanize.naturaldelta(min_sec)
        max_str = humanize.naturaldelta(max_sec)
        if min_str == max_str:
            return min_str
        return f"{min_str} to {max_str}"
```

**Step 5: Run test to verify it passes**

First install humanize in your environment:
```bash
cd python && uv pip install humanize
```

Run: `pytest python/tests/test_progress_utils.py::test_format_time_estimate_very_short -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_format_time_estimate_seconds -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_format_time_estimate_minutes -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_format_time_estimate_range -v`
Expected: PASS

**Step 6: Update pyproject.toml with humanize dependency**

Modify `python/pyproject.toml` to add `humanize` to the dependencies list.

Find the `[project.dependencies]` section and add:
```toml
"humanize>=4.0.0",
```

**Step 7: Commit**

```bash
git add python/pdstools/utils/progress_utils.py python/tests/test_progress_utils.py python/pyproject.toml
git commit -m "feat(utils): add time formatting with humanize library"
```

---

## Task 3: Add sampling time estimation function

**Files:**
- Modify: `python/pdstools/utils/progress_utils.py`
- Modify: `python/tests/test_progress_utils.py`

**Step 1: Write the failing test for estimate_sampling_time**

Add to `python/tests/test_progress_utils.py`:

```python
from pdstools.utils.progress_utils import estimate_sampling_time


def test_estimate_sampling_time_small():
    """Test estimation for small dataset."""
    # 10k rows, sampling 5k
    min_sec, max_sec = estimate_sampling_time(10_000, 5_000)
    assert min_sec < max_sec
    assert max_sec < 10  # Should be fast


def test_estimate_sampling_time_large():
    """Test estimation for large dataset."""
    # 10M rows, sampling 50k
    min_sec, max_sec = estimate_sampling_time(10_000_000, 50_000)
    assert min_sec < max_sec
    assert min_sec > 0


def test_estimate_sampling_time_no_sampling():
    """Test when no sampling needed (data smaller than sample size)."""
    min_sec, max_sec = estimate_sampling_time(1_000, 50_000)
    assert min_sec < max_sec
    assert max_sec < 1  # Very fast
```

**Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_progress_utils.py::test_estimate_sampling_time_small -v`
Expected: FAIL with "ImportError: cannot import name 'estimate_sampling_time'"

**Step 3: Write minimal implementation**

Add to `python/pdstools/utils/progress_utils.py`:

```python
def estimate_sampling_time(
    total_rows: int, sample_size: int
) -> tuple[float, float]:
    """Estimate time for sampling operations based on dataset size.

    Parameters
    ----------
    total_rows : int
        Total number of rows in the dataset
    sample_size : int
        Target sample size

    Returns
    -------
    tuple[float, float]
        (min_seconds, max_seconds) for a range estimate

    Examples
    --------
    >>> min_time, max_time = estimate_sampling_time(1_000_000, 50_000)
    >>> min_time < max_time
    True
    """
    # Calibrated sampling speeds (rows/second)
    # Based on Polars hash-based sampling performance
    FAST_ROWS_PER_SEC = 1_000_000  # Good CPU, in-memory data
    SLOW_ROWS_PER_SEC = 100_000  # Slower system or disk-based

    # If no sampling needed (data smaller than sample), very fast
    if total_rows <= sample_size:
        return (0.1, 0.5)

    # Estimate based on total rows (not sample size) since we scan all data
    min_time = total_rows / FAST_ROWS_PER_SEC
    max_time = total_rows / SLOW_ROWS_PER_SEC

    return (min_time, max_time)
```

**Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_progress_utils.py::test_estimate_sampling_time_small -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_estimate_sampling_time_large -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_estimate_sampling_time_no_sampling -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/pdstools/utils/progress_utils.py python/tests/test_progress_utils.py
git commit -m "feat(utils): add sampling time estimation function"
```

---

## Task 4: Update handle_data_path to show extraction time estimates

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/da_streamlit_utils.py:432-442`
- Reference: Design doc section on "Integration Points"

**Step 1: Read current implementation**

Read `python/pdstools/app/decision_analyzer/da_streamlit_utils.py` lines 432-442 to see current zip handling code.

**Step 2: Update with time estimation**

Modify the zip file handling section in `handle_data_path()`:

```python
# Single zip file: extract first, then read the extracted contents
if p.is_file() and p.suffix.lower() == ".zip":
    import tempfile
    import zipfile

    # Add time estimation
    try:
        from pdstools.utils.progress_utils import (
            estimate_extraction_time,
            format_time_estimate,
        )

        file_size = p.stat().st_size
        min_time, max_time = estimate_extraction_time(file_size)
        time_msg = format_time_estimate(min_time, max_time)
        spinner_msg = f"Extracting archive... (estimated: {time_msg})"
    except Exception:
        # Fall back to simple message if estimation fails
        spinner_msg = "Extracting archive..."

    tmp_dir = tempfile.mkdtemp(prefix="da_path_")
    with st.spinner(spinner_msg):
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp_dir)
    return read_data(tmp_dir)
```

**Step 3: Manual test**

Test manually with a real zip file:
1. Run the Decision Analyzer app
2. Use `--data-path` flag pointing to a zip file
3. Verify the spinner shows "Extracting archive... (estimated: X to Y)"
4. Verify extraction completes successfully

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/da_streamlit_utils.py
git commit -m "feat(decision_analyzer): add time estimates for zip extraction"
```

---

## Task 5: Update _read_uploaded_zip to show extraction time estimates

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/da_streamlit_utils.py:445-480`
- Reference: Similar to Task 4 but for uploaded files

**Step 1: Read current implementation**

Read `python/pdstools/app/decision_analyzer/da_streamlit_utils.py` lines 445-480 to see the `_read_uploaded_zip` function.

**Step 2: Add time estimation for uploaded files**

Find where the extraction happens in `_read_uploaded_zip()` and add time estimation:

Look for the `zf.extractall(tmp_dir)` call and wrap it with a spinner that has time estimates.

The file buffer won't have a simple file size, but you can get it from:
```python
file_buffer.seek(0, 2)  # Seek to end
file_size = file_buffer.tell()
file_buffer.seek(0)  # Seek back to start
```

Add time estimation similar to Task 4:

```python
# Before extractall, estimate time
try:
    from pdstools.utils.progress_utils import (
        estimate_extraction_time,
        format_time_estimate,
    )

    file_buffer.seek(0, 2)
    file_size = file_buffer.tell()
    file_buffer.seek(0)

    min_time, max_time = estimate_extraction_time(file_size)
    time_msg = format_time_estimate(min_time, max_time)
    spinner_msg = f"Extracting uploaded archive... (estimated: {time_msg})"
except Exception:
    spinner_msg = "Extracting uploaded archive..."

# Then use spinner_msg with st.spinner()
```

**Step 3: Manual test**

Test manually with uploaded zip file:
1. Run the Decision Analyzer app
2. Upload a zip file via the UI
3. Verify the spinner shows time estimate
4. Verify extraction completes successfully

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/da_streamlit_utils.py
git commit -m "feat(decision_analyzer): add time estimates for uploaded zip extraction"
```

---

## Task 6: Update sampling operation to show time estimates

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/Home.py:110-125`
- Reference: Design doc section on sampling enhancement

**Step 1: Read current sampling code**

Read `python/pdstools/app/decision_analyzer/Home.py` lines 110-125 to understand the current sampling flow.

**Step 2: Add helper function to estimate row count**

Before the sampling block, add a helper function or inline code to estimate row count:

```python
def _estimate_row_count(df: pl.LazyFrame) -> int | None:
    """Attempt to estimate row count from lazy frame without collecting.

    Returns None if estimation not possible.
    """
    try:
        # Try to get count from metadata (works for some formats)
        return df.select(pl.len()).collect().item()
    except Exception:
        # If that fails, we can't estimate without collecting
        return None
```

**Step 3: Update sampling message with time estimate**

Modify the sampling block around line 110-125:

```python
if sample_limit_raw:
    # Sampling mode
    try:
        sample_kwargs = parse_sample_flag(sample_limit_raw)
    except ValueError as e:
        st.error(f"Invalid --sample value: {e}")
        st.stop()

    # Build sampling message with time estimate if possible
    try:
        from pdstools.utils.progress_utils import (
            estimate_sampling_time,
            format_time_estimate,
        )

        # Try to estimate row count
        row_count = _estimate_row_count(raw_data)

        if row_count is not None:
            target_n = sample_kwargs.get("n", int(row_count * sample_kwargs.get("fraction", 1.0)))
            min_time, max_time = estimate_sampling_time(row_count, target_n)
            time_msg = format_time_estimate(min_time, max_time)

            if "fraction" in sample_kwargs:
                sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the interactions... (estimated: {time_msg})"
            elif "n" in sample_kwargs:
                sampling_msg = f"Sampling {sample_kwargs['n']:,} interactions... (estimated: {time_msg})"
            else:
                sampling_msg = f"Sampling interactions... (estimated: {time_msg})"
        else:
            # Fall back to simple message if can't estimate row count
            if "fraction" in sample_kwargs:
                sampling_msg = f"Sampling {sample_kwargs['fraction'] * 100:.0f}% of the interactions…"
            elif "n" in sample_kwargs:
                sampling_msg = f"Sampling {sample_kwargs['n']:,} interactions…"
            else:
                sampling_msg = "Sampling interactions…"
    except Exception:
        # Fall back to existing simple messages
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
```

**Step 4: Manual test**

Test manually with sampling:
1. Run the Decision Analyzer app with `--sample 10000` or similar
2. Verify the spinner shows time estimate
3. Verify sampling completes successfully

**Step 5: Commit**

```bash
git add python/pdstools/app/decision_analyzer/Home.py
git commit -m "feat(decision_analyzer): add time estimates for sampling operations"
```

---

## Task 7: Manual calibration and testing

**Files:**
- Test: Manual testing with real datasets
- Potentially modify: `python/pdstools/utils/progress_utils.py` (adjust speed constants)

**Step 1: Test with real large file (4.5 GB NAB dataset)**

1. Run Decision Analyzer with the 4.5 GB NAB zip file
2. Note the actual extraction time
3. Compare with the estimated time range
4. Record results

**Step 2: Evaluate estimation accuracy**

Check if the actual time falls within the estimated range:
- If actual time is significantly faster than min estimate: increase FAST_SPEED_MBS
- If actual time is significantly slower than max estimate: decrease SLOW_SPEED_MBS
- Aim for actual time to fall within the estimated range in most cases

**Step 3: Adjust speed constants if needed**

If estimates are off, update the constants in `progress_utils.py`:

```python
# For extraction:
FAST_SPEED_MBS = 100  # Adjust based on testing
SLOW_SPEED_MBS = 30   # Adjust based on testing

# For sampling:
FAST_ROWS_PER_SEC = 1_000_000  # Adjust based on testing
SLOW_ROWS_PER_SEC = 100_000    # Adjust based on testing
```

**Step 4: Test with various file sizes**

Test with:
- Small file (~10 MB)
- Medium file (~500 MB)
- Large file (~4.5 GB)

Verify messages are appropriate for each size.

**Step 5: Document calibration results**

Add comment in `progress_utils.py` documenting the calibration:

```python
# Calibrated extraction speeds (MB/s)
# Based on testing with:
# - 4.5 GB NAB dataset on MacBook Pro M1
# - Actual extraction time: X minutes
# - Estimated range: Y to Z minutes
FAST_SPEED_MBS = 100  # SSD with good CPU
SLOW_SPEED_MBS = 30   # HDD or compressed data
```

**Step 6: Commit if adjustments made**

```bash
git add python/pdstools/utils/progress_utils.py
git commit -m "chore(utils): calibrate time estimation constants based on testing"
```

---

## Task 8: Add integration test for error handling

**Files:**
- Modify: `python/tests/test_progress_utils.py`

**Step 1: Write test for fallback behavior**

Add to `python/tests/test_progress_utils.py`:

```python
def test_format_time_estimate_without_humanize(monkeypatch):
    """Test fallback when humanize not available."""
    # Mock humanize import failure
    import sys
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "humanize":
            raise ImportError("humanize not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Should still work with fallback
    result = format_time_estimate(30, 45)
    assert "45" in result or "second" in result.lower()

    result = format_time_estimate(120, 180)
    assert "minute" in result.lower()


def test_estimate_extraction_time_zero_size():
    """Test with zero-byte file."""
    min_sec, max_sec = estimate_extraction_time(0)
    assert min_sec == 0
    assert max_sec == 0


def test_estimate_sampling_time_zero_rows():
    """Test with empty dataset."""
    min_sec, max_sec = estimate_sampling_time(0, 1000)
    assert min_sec >= 0
    assert max_sec >= 0
```

**Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_progress_utils.py::test_format_time_estimate_without_humanize -v`
Expected: May pass already if fallback logic exists, or fail if not

**Step 3: Ensure fallback logic exists**

Verify `format_time_estimate()` has the try/except ImportError block for humanize.
If not, add it.

**Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_progress_utils.py::test_format_time_estimate_without_humanize -v`
Expected: PASS

Run: `pytest python/tests/test_progress_utils.py::test_estimate_extraction_time_zero_size -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/tests/test_progress_utils.py
git commit -m "test(utils): add integration tests for error handling"
```

---

## Task 9: Run full test suite

**Step 1: Run all progress_utils tests**

Run: `pytest python/tests/test_progress_utils.py -v`
Expected: All tests PASS

**Step 2: Run broader test suite**

Run tests for affected modules:
```bash
pytest python/tests/ -v -k "decision_analyzer or utils"
```

Expected: All existing tests still pass (no regressions)

**Step 3: Check test coverage for new module**

Run: `pytest python/tests/test_progress_utils.py --cov=pdstools.utils.progress_utils --cov-report=term-missing`

Expected: High coverage (>90%) for the new module

**Step 4: Fix any failing tests**

If any tests fail, investigate and fix:
- Check if imports are correct
- Verify fallback logic works
- Ensure no breaking changes

---

## Task 10: Update documentation

**Files:**
- Create or modify: User-facing docs if needed
- Modify: `python/pdstools/utils/progress_utils.py` (ensure docstrings are complete)

**Step 1: Review docstrings**

Ensure all functions in `progress_utils.py` have:
- Clear description
- Parameters documented with types
- Returns documented with types
- Examples in docstring

**Step 2: Add module-level docstring**

Add to top of `progress_utils.py`:

```python
"""Utilities for progress feedback and time estimation.

This module provides functions to estimate operation times and format them
in user-friendly ways. Used primarily by the Decision Analyzer Streamlit
app to show progress feedback for long-running operations like:

- Extracting large zip archives
- Sampling large datasets

The estimates are based on calibrated speeds and provide ranges to account
for system variability.
"""
```

**Step 3: Commit**

```bash
git add python/pdstools/utils/progress_utils.py
git commit -m "docs(utils): improve docstrings for progress utilities"
```

---

## Task 11: Final verification and cleanup

**Step 1: Verify all files are properly formatted**

Run: `ruff format python/pdstools/utils/progress_utils.py python/tests/test_progress_utils.py`

Run: `ruff check python/pdstools/utils/progress_utils.py python/tests/test_progress_utils.py`

Fix any issues reported.

**Step 2: Test the complete feature end-to-end**

1. Start Decision Analyzer with a large zip file
2. Verify extraction shows time estimate
3. Verify estimate is reasonably accurate
4. Test with sampling flag
5. Verify sampling shows time estimate

**Step 3: Review all commits**

Run: `git log --oneline -20`

Verify commit messages are clear and follow conventions.

**Step 4: Push changes (if appropriate)**

If working in a branch and ready to share:
```bash
git push origin HEAD
```

---

## Testing Checklist

- [ ] Unit tests for `estimate_extraction_time()` pass
- [ ] Unit tests for `format_time_estimate()` pass
- [ ] Unit tests for `estimate_sampling_time()` pass
- [ ] Integration tests for error handling pass
- [ ] Manual test with 4.5 GB NAB file shows reasonable estimate
- [ ] Manual test with small file (<10 MB) works
- [ ] Manual test with uploaded zip file works
- [ ] Manual test with sampling flag works
- [ ] No regressions in existing tests
- [ ] Time estimates fall within 50% accuracy for typical cases

## Notes

- Time estimation constants may need adjustment based on different hardware
- The `humanize` library is well-maintained but fallback logic ensures graceful degradation
- If row count estimation is too slow for large lazy frames, it will fall back to simple messages
- All changes are backward compatible - existing functionality unchanged

## Future Enhancements (Out of Scope)

- Real-time progress bars with actual extraction progress
- Adaptive calibration that learns from past loads
- CLI progress feedback
- Progress for other operations (directory scanning, initial processing)
- Cancellation support for long operations

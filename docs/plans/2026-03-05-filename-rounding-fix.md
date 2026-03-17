# Filename Rounding Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `format_count_for_filename()` to properly round numbers instead of producing scientific notation in filenames.

**Architecture:** Modify the formatting logic to round values first, then check thresholds for unit transitions. When rounded value reaches 100+ in current unit, transition to next unit for cleaner output.

**Tech Stack:** Python 3.10+, pytest

---

## Task 1: Write Failing Test for Rounding Edge Cases

**Files:**
- Modify: `python/tests/test_da_utils.py:125-136`

### Step 1: Add new test function after existing edge cases test

Insert after line 136 in `python/tests/test_da_utils.py`:

```python
def test_format_count_rounding_to_unit_boundaries():
    """Test that values rounding to 100+ transition to next unit cleanly."""
    from pdstools.decision_analyzer.utils import format_count_for_filename

    # Values that round to 100k should display as 100k
    assert format_count_for_filename(99500) == "100k"
    assert format_count_for_filename(99900) == "100k"

    # Values that round to 1000k should transition to 1M
    assert format_count_for_filename(999500) == "1M"
    assert format_count_for_filename(999900) == "1M"

    # Verify no scientific notation in output
    result = format_count_for_filename(99500)
    assert "e" not in result.lower(), f"Got scientific notation: {result}"
```

### Step 2: Run test to verify it fails

Run: `pytest python/tests/test_da_utils.py::test_format_count_rounding_to_unit_boundaries -v`

**Expected output:** FAIL with assertion errors showing `1e+02k` instead of `100k`

### Step 3: Commit the failing test

```bash
git add python/tests/test_da_utils.py
git commit -m "test(decision_analyzer): add failing test for filename rounding bug

Tests for values that should round to unit boundaries but currently
produce scientific notation (e.g., 99,500 → 1e+02k instead of 100k).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Fix format_count_for_filename Implementation

**Files:**
- Modify: `python/pdstools/decision_analyzer/utils.py:808-834`

### Step 1: Replace the function implementation

Replace lines 808-834 in `python/pdstools/decision_analyzer/utils.py` with:

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
        rounded = round(value)

        if rounded >= 100:
            # Transition to millions (will be handled by next elif)
            # Round to nearest million boundary to avoid recursion issues
            millions = count / 1_000_000
            rounded_m = round(millions)
            if rounded_m >= 100:
                return f"{rounded_m}M"
            elif rounded_m >= 10:
                return f"{rounded_m}M"
            else:
                return f"{millions:.1f}M".rstrip('0').rstrip('.')
        elif rounded >= 10:
            return f"{rounded}k"
        else:
            # Use 1 decimal for < 10
            formatted = f"{value:.1f}k"
            return formatted.rstrip('0').rstrip('.')
    elif count < 1_000_000_000:
        # Millions
        value = count / 1_000_000
        rounded = round(value)

        if rounded >= 100:
            # Transition to billions
            billions = count / 1_000_000_000
            rounded_b = round(billions)
            if rounded_b >= 100:
                return f"{rounded_b}B"
            elif rounded_b >= 10:
                return f"{rounded_b}B"
            else:
                return f"{billions:.1f}B".rstrip('0').rstrip('.')
        elif rounded >= 10:
            return f"{rounded}M"
        else:
            formatted = f"{value:.1f}M"
            return formatted.rstrip('0').rstrip('.')
    else:
        # Billions
        value = count / 1_000_000_000
        rounded = round(value)

        if rounded >= 10:
            return f"{rounded}B"
        else:
            formatted = f"{value:.1f}B"
            return formatted.rstrip('0').rstrip('.')
```

### Step 2: Run new test to verify it passes

Run: `pytest python/tests/test_da_utils.py::test_format_count_rounding_to_unit_boundaries -v`

**Expected output:** PASS

### Step 3: Verify no scientific notation with manual test

Run:
```bash
python3 -c "
import sys
sys.path.insert(0, 'python')
from pdstools.decision_analyzer.utils import format_count_for_filename

test_cases = [99400, 99500, 99900, 999500, 999900]
for count in test_cases:
    result = format_count_for_filename(count)
    print(f'{count:>7} → {result}')
    assert 'e' not in result.lower(), f'Scientific notation in {result}'
print('✓ No scientific notation found')
"
```

**Expected output:**
```
  99400 → 99k
  99500 → 100k
  99900 → 100k
 999500 → 1M
 999900 → 1M
✓ No scientific notation found
```

### Step 4: Commit the implementation

```bash
git add python/pdstools/decision_analyzer/utils.py
git commit -m "fix(decision_analyzer): fix rounding in filename formatting

Fixes format_count_for_filename to properly round numbers instead of
producing scientific notation. Values like 99,500 now produce '100k'
instead of '1e+02k'.

Changes:
- Round values before checking thresholds
- Transition to next unit when rounded value >= 100
- Use 1 decimal place for values < 10, strip trailing zeros
- No more .2g formatting that caused scientific notation

Fixes the bug where cached/sampled file names contained scientific
notation like 'decision_analyzer_sample_1e+02k.parquet'.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Existing Boundary Tests

**Files:**
- Modify: `python/tests/test_da_utils.py:105,114`

### Step 1: Update line 105 (thousands boundary test)

Change line 105 from:
```python
assert format_count_for_filename(999999) == "1000k"
```

To:
```python
assert format_count_for_filename(999999) == "1M"
```

### Step 2: Update line 114 (millions boundary test)

Change line 114 from:
```python
assert format_count_for_filename(999999999) == "1000M"
```

To:
```python
assert format_count_for_filename(999999999) == "1B"
```

### Step 3: Run updated tests to verify they pass

Run: `pytest python/tests/test_da_utils.py::test_format_count_for_filename_thousands -v`

**Expected output:** PASS

Run: `pytest python/tests/test_da_utils.py::test_format_count_for_filename_millions -v`

**Expected output:** PASS

### Step 4: Commit the test updates

```bash
git add python/tests/test_da_utils.py
git commit -m "test(decision_analyzer): update boundary tests for cleaner unit transitions

Updates tests to expect cleaner unit transitions:
- 999,999 now formats as '1M' instead of '1000k'
- 999,999,999 now formats as '1B' instead of '1000M'

This matches the improved formatting behavior where values rounding
to 1000+ in a unit transition to the next unit.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Run Full Test Suite

**Files:**
- No modifications

### Step 1: Run all format_count_for_filename tests

Run: `pytest python/tests/test_da_utils.py -k format_count_for_filename -v`

**Expected output:** All tests PASS (7 tests total)

### Step 2: Run complete decision_analyzer utils test suite

Run: `pytest python/tests/test_da_utils.py -v`

**Expected output:** All tests PASS

### Step 3: Manual smoke test with real-world values

Run:
```bash
python3 -c "
import sys
sys.path.insert(0, 'python')
from pdstools.decision_analyzer.utils import format_count_for_filename

print('Smoke test - Real-world values:')
test_cases = [
    (1234, '1.2k'),
    (49000, '49k'),
    (87432, '87k'),
    (99500, '100k'),
    (100000, '100k'),
    (999999, '1M'),
    (1234567, '1.2M'),
]

all_pass = True
for count, expected in test_cases:
    result = format_count_for_filename(count)
    status = '✓' if result == expected else '✗'
    print(f'{status} {count:>9} → {result:>6} (expected {expected})')
    if result != expected:
        all_pass = False
    if 'e' in result.lower():
        print(f'  ERROR: Scientific notation found!')
        all_pass = False

if all_pass:
    print('\n✓ All smoke tests passed!')
else:
    print('\n✗ Some tests failed!')
    sys.exit(1)
"
```

**Expected output:**
```
Smoke test - Real-world values:
✓      1234 →   1.2k (expected 1.2k)
✓     49000 →    49k (expected 49k)
✓     87432 →    87k (expected 87k)
✓     99500 →   100k (expected 100k)
✓    100000 →   100k (expected 100k)
✓    999999 →     1M (expected 1M)
✓   1234567 →   1.2M (expected 1.2M)

✓ All smoke tests passed!
```

### Step 4: Verify no regressions in related code

Run: `pytest python/tests/test_da_utils.py::test_create_sampled_data -v`

**Expected output:** PASS (verifies that file creation using format_count_for_filename still works)

---

## Success Criteria

After completing all tasks:

- [ ] No scientific notation (`1e+02k`) in any filename output
- [ ] Values like 99,500 produce `100k` instead of `1e+02k`
- [ ] Boundary transitions are cleaner (999,999 → `1M` not `1000k`)
- [ ] All 7 format_count_for_filename tests pass
- [ ] Complete test_da_utils.py suite passes
- [ ] Manual smoke tests confirm expected behavior
- [ ] No regressions in file creation functionality

## Notes

- The fix addresses a critical bug where scientific notation made filenames unusable
- The improved unit transitions (e.g., `1M` instead of `1000k`) are a welcome side effect
- This is not a breaking change - it fixes buggy behavior
- No API changes, only improved output formatting

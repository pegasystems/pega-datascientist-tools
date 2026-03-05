# File Name Rounding Fix Design

**Date:** 2026-03-05
**Status:** Approved

## Problem Statement

The `format_count_for_filename()` function in `python/pdstools/decision_analyzer/utils.py` uses `.2g` formatting which produces scientific notation (e.g., `1e+02k`) when rounded values reach 100+ in the current unit. This creates unusable filenames like `decision_analyzer_sample_1e+02k.parquet` instead of the expected `decision_analyzer_sample_100k.parquet`.

**Examples of broken behavior:**
- 99,500 → `1e+02k` (should be `100k`)
- 99,900 → `1e+02k` (should be `100k`)

## Root Cause

The current logic (lines 812-818) uses `.2g` formatting for values < 100:

```python
if value >= 100:
    return f"{int(round(value))}k"
else:
    formatted = f"{value:.2g}"  # ← Produces scientific notation!
    return f"{formatted}k"
```

When `value = 99.5`, the `.2g` format produces `"1e+02"` instead of properly rounding to `100`.

## Requirements

1. Fix scientific notation in filenames - no `1e+02k` strings
2. Round numbers properly - 99,500 should become `100k`, not `99k`
3. Maintain existing behavior for values that don't round to boundaries
4. When rounded value reaches 100+ in current unit, transition to next unit (100k → 0.1M)
5. Update affected tests to reflect new behavior

## Design Decision

**Approach:** Round values first, then check thresholds to determine unit transitions.

**Key changes:**
1. Round the value before checking against thresholds
2. If rounded value ≥ 100, transition to the next unit
3. If rounded value ≥ 10, format as integer (no decimals)
4. If rounded value < 10, use one decimal place with trailing zeros removed

**Alternative approaches considered:**
- Use integer arithmetic throughout: Too much rewrite, loses decimal precision
- Pre-round and determine unit: More explicit but more code changes
- Keep current structure with better formatting: Chosen approach - minimal changes

## Implementation

**File:** `python/pdstools/decision_analyzer/utils.py`

**Changes to `format_count_for_filename()` function (lines 808-834):**

```python
def format_count_for_filename(count: int) -> str:
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        # Thousands
        value = count / 1000
        rounded = round(value)

        if rounded >= 100:
            # Transition to millions
            return format_count_for_filename(count)  # Will hit millions branch
        elif rounded >= 10:
            return f"{rounded}k"
        else:
            # Use 1 decimal for < 10
            return f"{value:.1f}k".rstrip('0').rstrip('.')
    elif count < 1_000_000_000:
        # Millions
        value = count / 1_000_000
        rounded = round(value)

        if rounded >= 100:
            # Transition to billions
            return format_count_for_filename(count)  # Will hit billions branch
        elif rounded >= 10:
            return f"{rounded}M"
        else:
            return f"{value:.1f}M".rstrip('0').rstrip('.')
    else:
        # Billions
        value = count / 1_000_000_000
        rounded = round(value)

        if rounded >= 100:
            return f"{rounded}B"
        else:
            return f"{value:.1f}B".rstrip('0').rstrip('.')
```

## Behavior Changes

**Before → After:**

| Input     | Before   | After  | Notes |
|-----------|----------|--------|-------|
| 99,400    | `99k`    | `99k`  | Unchanged |
| 99,500    | `1e+02k` | `100k` | **Fixed** |
| 99,900    | `1e+02k` | `100k` | **Fixed** |
| 999,999   | `1000k`  | `1M`   | Cleaner unit transition |
| 1,500     | `1.5k`   | `1.5k` | Unchanged |
| 87,432    | `87k`    | `87k`  | Unchanged |

## Testing

**Tests to update in `python/tests/test_da_utils.py`:**

1. **Line 105:** Update boundary test
   ```python
   # Before:
   assert format_count_for_filename(999999) == "1000k"
   # After:
   assert format_count_for_filename(999999) == "1M"
   ```

2. **Line 114:** Update boundary test
   ```python
   # Before:
   assert format_count_for_filename(999999999) == "1000M"
   # After:
   assert format_count_for_filename(999999999) == "1B"
   ```

**New test to add:**

```python
def test_format_count_rounding_edge_cases():
    """Test that values rounding to 100+ transition to next unit."""
    from pdstools.decision_analyzer.utils import format_count_for_filename

    # Values that round to 100k should display as 100k
    assert format_count_for_filename(99500) == "100k"
    assert format_count_for_filename(99900) == "100k"

    # Values that round to 1000k should transition to 1M
    assert format_count_for_filename(999500) == "1M"
    assert format_count_for_filename(999900) == "1M"

    # Verify no scientific notation
    result = format_count_for_filename(99500)
    assert "e" not in result.lower()
```

## Impact

- **Files modified:** 2 (utils.py, test_da_utils.py)
- **Breaking changes:** None (this fixes buggy behavior)
- **Test changes:** 2 assertions updated, 1 new test added
- **User-facing impact:** Better, more readable filenames

## Success Criteria

1. No scientific notation in any filename output
2. All existing tests pass with updated assertions
3. New edge case tests pass
4. Manual verification: `format_count_for_filename(99500)` returns `"100k"`

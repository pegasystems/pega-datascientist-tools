# Test Coverage Needed for HealthCheck Refactoring

**Date:** 2026-03-17
**Branch:** `refactor/healthcheck-delegate-to-library`
**Issue:** Code coverage falls below 80% threshold

## New Methods Requiring Tests

### 1. ADMDatamart.get_last_data_for_report()
**File:** `python/pdstools/adm/ADMDatamart.py` (lines 757-795)
**Purpose:** Get the last snapshot of data formatted for report display

**Test cases needed:**
- Returns DataFrame with correct schema
- Fills null values with "NA" for string columns
- Fills nan/null with 0 for SuccessRate and Performance
- Creates "Channel/Direction" concatenated column
- Casts categorical columns to string

**Test file:** `python/tests/test_ADMDatamart.py`

### 2. Aggregates.summary_by_channel() - format_flags parameter
**File:** `python/pdstools/adm/Aggregates.py`
**Purpose:** Format boolean flags (usesNBAD, usesAGB) as "Yes"/"No"/"?" strings

**Test cases needed:**
- `format_flags=False` returns original boolean values
- `format_flags=True` converts True→"Yes", False→"No", null→"?"
- Works for both usesNBAD and usesAGB columns

**Test file:** `python/tests/test_Aggregates.py`

### 3. Plots.gains_chart()
**File:** `python/pdstools/adm/Plots.py` (lines 1181-1292)
**Purpose:** Generate a gains chart showing cumulative distribution of a metric

**Test cases needed:**
- Creates figure with correct trace type
- `return_df=True` returns LazyFrame
- `by` parameter creates multiple traces
- `index` parameter sorts data correctly
- `query` parameter filters data
- Validates value/index columns exist in data

**Test file:** `python/tests/test_Plots.py`

### 4. Plots.performance_volume_distribution()
**File:** `python/pdstools/adm/Plots.py` (lines 1294-1437)
**Purpose:** Generate a performance vs volume distribution chart

**Test cases needed:**
- Creates figure with correct layout
- `return_df=True` returns LazyFrame
- `bin_width` parameter affects histogram binning
- `by` parameter creates facets
- `query` parameter filters data
- Handles models with zero responses

**Test file:** `python/tests/test_Plots.py`

### 5. report_utils.gains_table()
**File:** `python/pdstools/utils/report_utils.py` (lines 1003-1085)
**Purpose:** Calculate cumulative gains for visualization

**Test cases needed:**
- Sorts by value column correctly
- Calculates cumulative sums accurately
- Handles `by` parameter for grouping
- Handles `index` parameter for sorting
- Returns correct schema (Bin, Count, Value, CumulativeValue, CumulativePercent)
- Edge case: single row input
- Edge case: zero values

**Test file:** `python/tests/test_report_utils.py` (may need to create)

## Test Implementation Plan

1. **Priority 1 (Critical for coverage):**
   - `gains_chart()` - likely not covered at all
   - `performance_volume_distribution()` - likely not covered at all
   - `gains_table()` - utility function, likely not covered

2. **Priority 2 (Medium):**
   - `get_last_data_for_report()` - simple formatting, quick to test
   - `format_flags` parameter - small addition to existing method

3. **Test Strategy:**
   - Use `return_df=True` pattern for plot methods to test data logic
   - Use fixtures from existing tests (e.g., `cdh_sample()`)
   - Add parametrized tests for different configurations

## Example Test Structure

```python
def test_gains_chart_basic(datamart_instance):
    """Test basic gains chart creation."""
    fig = datamart_instance.plot.gains_chart(
        value="ResponseCount",
        index="ModelID"
    )
    assert isinstance(fig, Figure)

def test_gains_chart_return_df(datamart_instance):
    """Test gains chart data return."""
    df = datamart_instance.plot.gains_chart(
        value="ResponseCount",
        index="ModelID",
        return_df=True
    )
    assert isinstance(df, pl.LazyFrame)
    schema = df.collect_schema().names()
    assert "Bin" in schema
    assert "CumulativePercent" in schema

def test_gains_chart_with_grouping(datamart_instance):
    """Test gains chart with by parameter."""
    fig = datamart_instance.plot.gains_chart(
        value="ResponseCount",
        by="Channel"
    )
    assert isinstance(fig, Figure)
    # Should have multiple traces, one per Channel
    assert len(fig.data) > 1
```

## Running Tests

```bash
# Run specific test file
uv run pytest python/tests/test_Plots.py -v

# Run with coverage
uv run pytest python/tests --cov=python/pdstools --cov-report=term-missing

# Run only new tests
uv run pytest python/tests -k "gains_chart or performance_volume"
```

## Notes

- The batch script `generate_customer_healthchecks.py` is a utility script, not library code, so it doesn't need to meet the 80% threshold
- Focus tests on the library methods that are now being called by the report
- Use existing test fixtures and patterns from the codebase

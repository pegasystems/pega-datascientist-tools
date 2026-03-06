# Propensity Validation Design

**Date:** 2026-03-06
**Status:** Implemented

## Problem Statement

While investigating dataset propensity distributions, we discovered unusually high propensity values (double-digit percentages) in production data. Marketing interactions typically have propensities < 1%, so values > 10% are unusual and may indicate:
- Model calibration issues
- Different modeling approaches (high-intent channels)
- Data quality or extraction problems

Users need to be warned about these issues early in their analysis workflow.

## Context

- Propensity values represent probability of interaction (should be 0.0-1.0)
- Typical marketing propensities are < 1% for banner/email interactions
- Sample data investigation revealed 90.9% of records with propensities > 10%
- Current validation only checks for missing required columns
- No data quality checks beyond column presence validation

## Requirements

1. **Detect invalid propensities** - Values > 1.0 violate probability constraints
2. **Detect unusually high propensities** - Values > 10% are uncommon for typical marketing
3. **Non-blocking warnings** - Analysis should continue normally
4. **Informative messaging** - Explain the issue and potential causes
5. **Efficient validation** - Use sample data to avoid performance impact
6. **Handle edge cases** - Missing columns, null values, default values (0.5)
7. **Organized presentation** - Don't clutter the Home page

## Design Decision

### Approach: Cached Property Validation + Dedicated Data Quality Page

**Core validation:**
- Add `propensity_validation_warning` cached property to DecisionAnalyzer
- Two-tier warning system:
  - **Invalid (> 1.0)**: Priority warning - mathematically impossible
  - **High (> 10%)**: Warning when >5% of records affected - unusual but possibly valid
- Use sample data for efficiency (same pattern as `stages_with_propensity`)
- Compute during initialization, cache result

**UI presentation:**
- Create new **1_Data_Quality.py** page dedicated to all validation checks
- Show brief notification on Home page linking to Data Quality page
- Renumber all existing pages to accommodate new page
- Data Quality page positioned as first page after Home (high visibility)

**Alternative approaches considered:**

1. ❌ **Display warnings on Home page** - Would clutter data loading interface
2. ❌ **Modal popup warnings** - Disruptive to workflow
3. ❌ **Validation on demand** - Users might miss issues
4. ❌ **Add to existing pages** - No logical home for general data quality
5. ✅ **Dedicated Data Quality page** - Scalable, organized, prominent

## Implementation

### 1. Core Validation Logic

**File:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`

Add cached property after `stages_with_propensity` (line ~402):

```python
@cached_property
def propensity_validation_warning(self) -> str | None:
    """Validate propensity values and return warning message if issues detected.

    Checks for:
    1. Invalid propensities (> 1.0) - mathematically impossible
    2. Unusually high propensities (> 0.1) - uncommon for marketing

    Returns None if validation passes or propensity data unavailable.
    """
    # Edge cases: missing column, no stages with propensity
    if "Propensity" not in self.sample.collect_schema().names():
        return None
    if not self.stages_with_propensity:
        return None

    # Filter to stages with propensity, compute statistics
    # Check for > 1.0 (invalid) and > 0.1 (high)
    # Return formatted warning message or None
```

**Logic:**
- Filter sample data to stages with meaningful propensity
- Exclude null, NaN, Inf values
- Calculate: max, 95th percentile, count > 1.0, count > 0.1
- Build warning message with affected stages
- Only warn for high propensities if >5% of records affected

### 2. Data Quality Page

**File:** `python/pdstools/app/decision_analyzer/pages/1_Data_Quality.py` (NEW)

```python
# Display validation warnings
if da.validation_error:
    st.warning(f"**Column Validation Issue:**\n\n{da.validation_error}")

if hasattr(da, "propensity_validation_warning") and da.propensity_validation_warning:
    st.warning(da.propensity_validation_warning)

# If no warnings, show success message
if not warnings_found:
    st.success("✅ No data quality issues detected")

# Show data quality summary metrics
# - Extract type, propensity availability, interaction counts
```

**Content:**
- Validation warnings section (column + propensity)
- Data quality summary metrics
- Expandable info section explaining all checks
- Room for future checks (duplicates, temporal consistency)

### 3. Home Page Update

**File:** `python/pdstools/app/decision_analyzer/Home.py`

Update `_show_data_summary()` function:

```python
def _show_data_summary(da):
    """Display a summary banner for the loaded DecisionAnalyzer."""
    # Check for any warnings
    has_warnings = (
        da.validation_error or
        (hasattr(da, "propensity_validation_warning") and da.propensity_validation_warning)
    )

    if has_warnings:
        st.info("ℹ️ Data quality warnings detected. Please review the Data Quality page.")

    # ... rest of summary
```

### 4. Page Renumbering

All existing pages shift by 1:
- `1_Data_Quality.py` (NEW)
- `2_Global_Data_Filters.py` (was 1)
- `3_Overview.py` (was 2)
- `4_Action_Distribution.py` (was 3)
- ... through ...
- `12_About.py` (was 11)

## Warning Message Format

### Invalid Propensities (> 1.0)

```
⚠️ Invalid propensity values detected:

• {count} records ({pct}%) have propensities > 1.0
• Maximum propensity: {max}
• Affected stages: {stages}

Propensities should be between 0 and 1. Please check your model
calibration or data extraction process.
```

### High Propensities (> 10%)

```
ℹ️ Unusually high propensities detected:

• {pct}% of records have propensities > 10%
• 95th percentile propensity: {p95}%
• Affected stages: {stages}

This is unusual for typical marketing interactions (usually < 1%).
This might indicate:
• Different model calibration approach
• High-intent channels or contexts
• Potential data quality issues

Consider reviewing your model configuration if this seems unexpected.
```

## Edge Cases

1. **Missing Propensity column** - Return None (validation not applicable)
2. **All null values** - Return None (no propensity data to validate)
3. **All default values (0.5)** - Return None (no real propensity data)
4. **No stages with propensity** - Return None (use existing detection)
5. **NaN/Inf values** - Filter out before computing statistics
6. **Small percentage (< 5%) high** - Don't warn (acceptable outliers)

## Testing

### Unit Tests

**File:** `python/tests/test_DecisionAnalyzer.py`

Add test class `TestPropensityValidation`:

1. `test_sample_data_propensity_detection` - Validates with real data
2. `test_high_propensities_triggers_warning` - Modifies data to test high detection
3. `test_invalid_propensities_triggers_warning` - Tests > 1.0 detection
4. `test_missing_propensity_column_no_warning` - Tests missing column handling
5. `test_propensity_validation_warning_is_cached` - Verifies cached property

### Manual Testing

1. Load sample data with high propensities
2. Verify warning displays on Data Quality page
3. Verify Home page shows notification
4. Check warning message format and content
5. Verify navigation to Data Quality page works

## Impact

**Files Modified:**
- `python/pdstools/decision_analyzer/DecisionAnalyzer.py` - Core validation logic
- `python/pdstools/app/decision_analyzer/Home.py` - Notification update
- `python/tests/test_DecisionAnalyzer.py` - Unit tests

**Files Created:**
- `python/pdstools/app/decision_analyzer/pages/1_Data_Quality.py` - New page

**Files Renamed:**
- All numbered pages shifted by 1 (1→2, 2→3, ... 11→12)

**Breaking Changes:** None - purely additive feature

**Performance Impact:** Minimal - uses cached sample data

## Future Enhancements

The Data Quality page framework enables easy addition of:
- Duplicate interaction detection
- Temporal consistency checks (date ranges, gaps)
- Stage progression validation (logical flow)
- Missing value analysis across all columns
- Data freshness indicators
- Distribution anomaly detection

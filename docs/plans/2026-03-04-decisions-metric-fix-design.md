# Decisions Metric Fix - Design

**Date:** 2026-03-04 (retroactive documentation)

**Status:** ✅ Completed

## Overview

Changed the "Decisions" metric from counting action occurrences (rows) to counting unique customer interactions. This provides more accurate and intuitive business metrics throughout the Decision Analyzer app.

## Problem Statement

The original "Decisions" metric counted total action occurrences (rows in the filtered data), which led to inflated numbers when the same customer interaction had an action at multiple stages.

**Example Problem**:
- Customer Interaction 1 has Action A at both Applicability stage and Arbitration stage
- Old behavior: Action A "Decisions" = 2 (counted twice)
- Expected: Action A "Decisions" = 1 (one unique customer received this action)

**User Impact**:
- Action Distribution treemap hover tooltips showed inflated "Decisions" counts
- Funnel metrics were misleading when actions appeared at multiple stages
- Filter impact analysis double-counted the same customer interaction
- Business users couldn't answer: "How many unique customers received this offer?"

## Root Cause

The aggregation logic used `pl.len()` which counts all rows in a group, not unique interactions:

```python
# OLD - Counts rows (action occurrences)
.agg([
    pl.len().alias("Decisions"),  # ❌ Counts duplicates
    ...
])
```

## Solution Architecture

### Core Change: Store Interaction IDs

Instead of just counting rows, store the actual interaction IDs and count unique values.

**New Approach**:
1. During initial aggregation: Store interaction IDs as a list
2. Count unique IDs for "Decisions" metric
3. When aggregating across stages: Flatten, deduplicate, and count

### Two Aggregation Paths

#### Path 1: Filter View (Stage-Specific)

Used by: Action Distribution page, filter details

**Implementation**:
```python
# NEW - Store IDs and count unique
.agg([
    pl.col("Interaction ID").alias("Interaction_IDs"),  # Store as list
    pl.col("Interaction ID").n_unique().alias("Decisions"),  # Count unique
    ...
])
```

**Benefit**: Each action at each stage now shows unique customer count, not row count.

#### Path 2: Remaining View (Cross-Stage)

Used by: Funnel charts, global metrics

**Challenge**: Same interaction may have an action at Stage A and Stage B. When aggregating "remaining" across stages, we need to avoid counting that interaction twice.

**Implementation**:
```python
# NEW - Flatten lists and count unique across stages
.agg([
    pl.col("Interaction_IDs")
        .flatten()       # Combine all lists
        .unique()        # Remove duplicates
        .len()           # Count unique
        .alias("Decisions"),
    ...
])
```

**Benefit**: Funnel metrics now show unique customer counts even when actions span multiple stages.

## Changes by Function

### 1. `getPreaggregatedFilterView()`

**Purpose**: Pre-aggregate data filtered to specific stages

**Change**:
```python
# Before
.agg([pl.len().alias("Decisions")])

# After
.agg([
    pl.col("Interaction ID").alias("Interaction_IDs"),
    pl.col("Interaction ID").n_unique().alias("Decisions"),
])
```

**Impact**: Action Distribution treemap now shows unique customer count per action per stage.

### 2. `getPreaggregatedRemainingView()`

**Purpose**: Pre-aggregate data showing actions remaining at/after each stage

**Change**:
```python
# Before
.agg([pl.len().alias("Decisions")])

# After
.agg([
    pl.col("Interaction ID").alias("Interaction_IDs"),
    pl.col("Interaction ID").n_unique().alias("Decisions"),
])
```

**Impact**: Remaining view now tracks unique customers, not action occurrences.

### 3. `getFunnelData()` - Two Aggregations

#### 3a. Filtered Funnel
```python
# Before
pl.len().alias("Decisions")

# After
pl.col("Interaction_IDs").flatten().unique().len().alias("Decisions")
```

#### 3b. Remaining Funnel
```python
# Before
pl.len().alias("Decisions")

# After
pl.col("Interaction_IDs").flatten().unique().len().alias("Decisions")
```

**Impact**: Funnel charts show unique customer counts at each stage.

### 4. `getFilterComponentData()`

**Purpose**: Show impact of each filter component on actions

**Change**:
```python
# Before
pl.len().alias("Decisions")

# After
pl.col("Interaction_IDs").flatten().unique().len().alias("Decisions")
```

**Impact**: Filter impact analysis shows unique customers affected, not total action occurrences.

## Data Flow

```
Input: Raw decision data (many rows per interaction if action at multiple stages)
  ↓
Group by: [Action, Channel, Direction, Issue, Group, Stage]
  ↓
Store:
  - Interaction_IDs: List of interaction IDs in this group
  - Decisions: Count of unique interaction IDs (n_unique)
  ↓
When aggregating across stages:
  - Flatten all Interaction_IDs lists
  - Remove duplicates (unique)
  - Count remaining IDs
  ↓
Output: "Decisions" = Unique customers who received this action/filter combination
```

## Example

**Sample Data**:
| Interaction ID | Action | Stage        |
|----------------|--------|--------------|
| INT001         | A      | Applicability|
| INT001         | A      | Arbitration  |
| INT002         | A      | Applicability|

### Old Behavior:
```
Action A across all stages:
  Decisions = 3 (counted 3 rows)
```

### New Behavior:
```
Action A at Applicability:
  Decisions = 2 (INT001, INT002)

Action A at Arbitration:
  Decisions = 1 (INT001)

Action A across all stages:
  Decisions = 2 (INT001, INT002 - deduplicated)
```

## File Modified

- `python/pdstools/decision_analyzer/DecisionAnalyzer.py`
  - Modified `getPreaggregatedFilterView()`
  - Modified `getPreaggregatedRemainingView()`
  - Updated `getFunnelData()` (both filtered and remaining)
  - Updated `getFilterComponentData()`

## Testing

All 133 existing tests continued to pass with the new metric semantics. The API remained backward compatible - only the meaning of "Decisions" changed from "action occurrences" to "unique interactions".

**Manual Verification**:
- Action Distribution treemap: Hover tooltips show correct unique customer counts
- Funnel charts: No double-counting when actions span stages
- Filter impact: Shows unique customers affected per filter

## Commit

- `61e350b8`: Count unique interactions instead of action occurrences in "Decisions" metric

## Impact

### Accuracy
- ✅ "Decisions" now represents unique customers, not inflated row counts
- ✅ No double-counting when same interaction has action at multiple stages
- ✅ Metrics align with business user expectations

### User Experience
- ✅ Action Distribution tooltips show meaningful customer reach
- ✅ Funnel metrics show true customer drop-off
- ✅ Filter impact analysis shows unique customers affected

### Backward Compatibility
- ✅ API unchanged - same function signatures
- ✅ All existing tests pass
- ✅ Only semantic change (what "Decisions" counts)

## Semantic Note

This is a **metric definition change**, not a bug fix. The old behavior (counting rows) was internally consistent, but the new behavior (counting unique customers) is more aligned with business questions like:
- "How many customers saw this offer?"
- "How many unique customers does this filter affect?"
- "What's our customer reach for this action?"

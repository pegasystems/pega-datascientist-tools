# Channel Filter for Optionality Analysis - Design Document

**Date:** 2026-03-11
**Status:** Approved
**Author:** AI Assistant (Claude)

## Overview

Add a Channel/Direction selection dropdown to the Optionality Analysis page sidebar, allowing users to filter all analyses on that page to a specific channel combination. The filter state persists across page navigation and respects existing global filters.

## Requirements

### Functional Requirements

1. **Channel Selection UI**
   - Dropdown in the Optionality Analysis page sidebar
   - Shows Channel/Direction combinations (e.g., "Web/Inbound", "Email/Outbound")
   - Includes an "Any" option to show all channels (default)
   - Positioned below the Stage Granularity selector

2. **Filtering Behavior**
   - "Any" selected: Show aggregated data across all channels (current behavior)
   - Specific channel selected: Show only data for that Channel/Direction combination
   - Filter applies to all four visualizations on the Optionality Analysis page:
     - Optionality (propensity vs optionality)
     - Optionality Funnel
     - Optionality Trend
     - Offer Variation

3. **Global Filter Integration**
   - Channel selector respects existing global filters from Global Data Filters page
   - Only shows Channel/Direction combinations that exist after global filters are applied
   - "Any" option respects global channel filters (doesn't override them)

4. **State Persistence**
   - Selection persists when navigating away and returning to the page
   - Filter state is global (stored in session_state) but only used by Optionality page initially
   - Designed to be extensible to other pages in the future

5. **Data Compatibility**
   - V2 data (Decision Analyzer/EEV2): Show Channel/Direction combinations
   - V1 data (Explainability Extract): Show Channel only (Direction may be missing)
   - Handle missing channel data gracefully

### Non-Functional Requirements

1. **Performance**: Filtering should be responsive even with large datasets (50k+ interactions)
2. **Robustness**: Handle edge cases gracefully with user-friendly messages
3. **Maintainability**: Clean separation of concerns, extensible to other page-level filters
4. **Consistency**: Follow existing UI patterns and code conventions in the project

## Architecture & Data Flow

### Data Flow Diagram

```
Raw Data
  ↓
Global Filters (from Global Data Filters page)
  ↓
DecisionAnalyzer.decision_data
  ↓
Sampling (DecisionAnalyzer.sample)
  ↓
Page-Level Filters (Channel/Direction) ← NEW
  ↓
DecisionAnalyzer.filtered_sample ← NEW
  ↓
Visualizations
```

### Key Architectural Decisions

1. **Filter Application Point**: Page-level filters apply to the `sample` property (after global filters and sampling)
2. **Filter Storage**: Channel selection stored in `st.session_state.page_channel_filter`
3. **Filter Scope**: Filter state is global (persists), but only affects pages that explicitly use `filtered_sample`
4. **Backwards Compatibility**: Existing code using `.sample` continues to work unchanged

### Why This Layering?

- **Global filters** reduce the full dataset to relevant interactions
- **Sampling** makes intensive analysis tractable
- **Page filters** let users drill into specific channel/direction combinations without affecting other pages yet

## Implementation Design

### Approach: Filtered Sample Property

We implement a new `filtered_sample` property on DecisionAnalyzer that applies page-level filters on top of the base sample. This provides:
- Clean separation of concerns
- All analyses automatically get filtered data
- Extensible to other page-level filters
- Consistent API

### Component Changes

#### A. DecisionAnalyzer Class (`DecisionAnalyzer.py`)

Add a new property that applies page-level filters:

```python
@property
def filtered_sample(self):
    """Sample data with page-level filters applied.

    Reads filter expressions from st.session_state.page_channel_expr if available.
    Falls back to unfiltered sample if no page filters are set.

    Not cached as @cached_property because it depends on mutable session_state.
    Calling code should cache at page level if needed.

    Returns
    -------
    pl.LazyFrame
        Sampled data with page filters applied, or unfiltered sample if no filters.
    """
    try:
        import streamlit as st
        channel_expr = st.session_state.get("page_channel_expr", None)
        if channel_expr is not None:
            return apply_filter(self.sample, [channel_expr])
    except ImportError:
        pass  # Not in Streamlit context

    return self.sample
```

**Lines changed:** ~15 lines added

#### B. Streamlit Utilities (`da_streamlit_utils.py`)

Add helper functions for channel selection:

```python
def get_available_channel_directions(sample_df: pl.LazyFrame) -> list[str]:
    """Get list of Channel/Direction combinations from the sample data.

    Handles both v2 data (with Direction) and v1 data (Channel only).

    Parameters
    ----------
    sample_df : pl.LazyFrame
        The sample data (after global filters applied).

    Returns
    -------
    list[str]
        List of "Channel/Direction" strings for v2 data,
        or list of "Channel" strings for v1 data,
        or empty list if no channel data available.
    """
    schema = sample_df.collect_schema().names()

    if "Direction" in schema:
        # V2 data: Combine Channel + Direction
        combos = (
            sample_df
            .select(pl.struct("Channel", "Direction"))
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        return sorted([f"{c['Channel']}/{c['Direction']}" for c in combos])
    elif "Channel" in schema:
        # V1 data: Channel only
        return sorted(
            sample_df.select("Channel").unique().collect().to_series().to_list()
        )
    else:
        # No channel data at all
        return []


def channel_direction_selector():
    """Render channel/direction selector in sidebar.

    Stores selection in st.session_state.page_channel_filter (UI value)
    and st.session_state.page_channel_expr (Polars filter expression).

    Handles edge cases:
    - Resets to "Any" if previous selection no longer available
    - Shows warning if no channel data available
    - Updates filter expression when selection changes
    """
    da = st.session_state.decision_data

    # Get available channel/direction combinations (respects global filters)
    available = get_available_channel_directions(da.sample)

    if not available:
        st.warning("No channel data available with current global filters.")
        st.session_state.page_channel_filter = "Any"
        st.session_state.page_channel_expr = None
        return

    # Check if current selection is still valid
    current = st.session_state.get("page_channel_filter", "Any")
    if current != "Any" and current not in available:
        st.info(
            f"Previous selection '{current}' is no longer available. Reset to 'Any'."
        )
        current = "Any"
        st.session_state.page_channel_filter = "Any"
        st.session_state.page_channel_expr = None

    # Build options list
    options = ["Any"] + available

    # Render selectbox
    selected = st.selectbox(
        "Channel / Direction",
        options=options,
        index=options.index(current),
        key="_channel_direction_widget",
        on_change=_update_channel_filter,
        help="Filter analysis to a specific channel. 'Any' shows all channels that pass global filters.",
    )


def _update_channel_filter():
    """Callback to update filter expression when channel selection changes."""
    selected = st.session_state._channel_direction_widget
    st.session_state.page_channel_filter = selected

    if selected == "Any":
        st.session_state.page_channel_expr = None
    else:
        # Parse "Channel/Direction" or just "Channel"
        if "/" in selected:
            channel, direction = selected.split("/", 1)
            st.session_state.page_channel_expr = (
                (pl.col("Channel") == channel) & (pl.col("Direction") == direction)
            )
        else:
            # V1 data: channel only
            st.session_state.page_channel_expr = pl.col("Channel") == selected
```

**Lines changed:** ~80 lines added

#### C. Optionality Analysis Page (`pages/8_Optionality_Analysis.py`)

Changes needed:

1. Add channel selector to sidebar (after line 23)
2. Compute filtered data once at top of page
3. Replace `decision_data.sample` with filtered data in all 4 sections
4. Add empty data check

```python
"# Optionality Analysis"

"""
Analysis of the number of actions per customer. Do we have enough options for people? Global filters can
be applied like everywhere to look at e.g. a certain group of issues.
"""
ensure_data()
st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()  # NEW

# Compute filtered data once (NEW)
filtered_data = st.session_state.decision_data.filtered_sample

# Check for empty results (NEW)
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.count()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()

# Section 1: Optionality (CHANGED: use filtered_data)
with st.container(border=True):
    "## Optionality"
    st.caption(...)
    stage_selectbox(key="optionality_stage", default="Arbitration")
    st.plotly_chart(
        st.session_state.decision_data.plot.propensity_vs_optionality(
            stage=st.session_state.get("optionality_stage", "Arbitration"),
            df=filtered_data,  # CHANGED from decision_data.sample
        ),
        width="stretch",
    )
    st.caption(...)

# Section 2: Optionality Funnel (CHANGED: use filtered_data)
if st.session_state.decision_data.extract_type != "explainability_extract":
    with st.container(border=True):
        "## Optionality Funnel"
        "Distribution of Available action by Stage"
        if st.session_state.decision_data.extract_type == "decision_analyzer":
            st.plotly_chart(
                st.session_state.decision_data.plot.optionality_funnel(
                    df=filtered_data  # CHANGED from decision_data.sample
                ),
                width="stretch",
            )

# Section 3: Optionality Trend (CHANGED: use filtered_data)
with st.container(border=True):
    "## Optionality Trend"
    st.caption(...)
    optionality_data_with_trend_per_stage = (
        st.session_state.decision_data.get_optionality_data_with_trend(
            df=filtered_data  # CHANGED from decision_data.sample
        )
        .group_by(["day", st.session_state.decision_data.level])
        .agg(nOffers=pl.col("nOffers").max())
        .sort("day")
    )
    fig, warning = st.session_state.decision_data.plot.optionality_trend(
        optionality_data_with_trend_per_stage,
    )
    if warning is not None:
        st.warning(warning)
    st.plotly_chart(fig, width="stretch")

# Section 4: Offer Variation (already uses "Output" stage, no df parameter)
# No change needed - this section doesn't currently accept a df parameter
```

**Lines changed:** ~15 lines added/modified

### Summary of Changes

| File | Lines Added | Lines Modified | Complexity |
|------|-------------|----------------|------------|
| `DecisionAnalyzer.py` | ~15 | 0 | Low |
| `da_streamlit_utils.py` | ~80 | 0 | Medium |
| `8_Optionality_Analysis.py` | ~10 | ~5 | Low |
| **Total** | **~105** | **~5** | **Low-Medium** |

## Technical Implementation Details

### A. Caching Strategy

**Why not @cached_property?**
The `filtered_sample` depends on `st.session_state`, which can change between page renders. Using `@cached_property` would cache stale data.

**Solution: Regular @property**
```python
@property
def filtered_sample(self):
    # Not cached because it depends on mutable session_state
    # Calling code should cache at page level if needed
```

**Page-level caching (if needed):**
```python
# Pages can cache the filtered data locally if performance is an issue
if "optionality_filtered_data" not in st.session_state:
    st.session_state.optionality_filtered_data = decision_data.filtered_sample
```

### B. Filter State Management

Store these values in `st.session_state`:

1. **`page_channel_filter`** (str): UI widget value ("Any", "Web/Inbound", etc.)
   - What the user sees and selects
   - Persists across page navigation

2. **`page_channel_expr`** (pl.Expr | None): The Polars filter expression
   - `None` for "Any" (no filtering)
   - Polars expression like `(pl.col("Channel") == "Web") & (pl.col("Direction") == "Inbound")`
   - Used by `filtered_sample` property

3. **`_channel_direction_widget`** (str): Temporary widget key
   - Streamlit internal widget state
   - Cleared by `on_change` callback

### C. Edge Case Handling

#### Edge Case 1: Direction Column Missing (V1 Data)
**Scenario:** Explainability Extract (v1) data may not have Direction column

**Handling:**
```python
if "Direction" in schema:
    # V2: Use Channel/Direction combinations
    return [f"{c['Channel']}/{c['Direction']}" for c in combos]
elif "Channel" in schema:
    # V1: Use Channel only
    return list(unique_channels)
```

#### Edge Case 2: No Channel Data After Global Filters
**Scenario:** Global filters eliminate all data, no channels available

**Handling:**
```python
if not available:
    st.warning("No channel data available with current global filters.")
    st.session_state.page_channel_filter = "Any"
    st.session_state.page_channel_expr = None
    return
```

#### Edge Case 3: Selected Channel No Longer Available
**Scenario:** User selects "Web/Inbound", then applies global filter that excludes Web

**Handling:**
```python
if current != "Any" and current not in available:
    st.info(f"Previous selection '{current}' is no longer available. Reset to 'Any'.")
    current = "Any"
    st.session_state.page_channel_filter = "Any"
    st.session_state.page_channel_expr = None
```

#### Edge Case 4: Empty Filtered Results
**Scenario:** Channel filter returns zero interactions

**Handling:**
```python
if filtered_count == 0:
    st.warning(
        f"No data available for {st.session_state.page_channel_filter}. "
        "Try selecting 'Any' or adjusting global filters."
    )
    st.stop()  # Stop rendering the page
```

#### Edge Case 5: ImportError (Non-Streamlit Context)
**Scenario:** DecisionAnalyzer used outside Streamlit (e.g., in tests or notebooks)

**Handling:**
```python
try:
    import streamlit as st
    # Apply filters from session_state
except ImportError:
    pass  # Not in Streamlit context, return unfiltered sample
return self.sample
```

## User Experience & UI Details

### Sidebar Layout

```
┌─────────────────────────────┐
│ SIDEBAR                     │
├─────────────────────────────┤
│ Stage Granularity           │
│ ○ Stage Group  ○ Stage      │  ← Existing
├─────────────────────────────┤
│ Channel / Direction         │  ← NEW
│ ┌─────────────────────────┐ │
│ │ Any                   ▼ │ │
│ └─────────────────────────┘ │
│                             │
│ Options:                    │
│ - Any                       │
│ - Web / Inbound             │
│ - Web / Outbound            │
│ - Email / Outbound          │
│ - Mobile / Inbound          │
│ ...                         │
└─────────────────────────────┘
```

### Widget Configuration

- **Label:** "Channel / Direction"
- **Help text:** "Filter analysis to a specific channel. 'Any' shows all channels that pass global filters."
- **Default:** "Any"
- **Position:** Below Stage Granularity selector, top of sidebar
- **Style:** Standard Streamlit selectbox (consistent with existing selectors)

### User Feedback Messages

| Scenario | Message Type | Message Text |
|----------|-------------|-------------|
| Selection becomes invalid | `st.info()` | "Previous selection 'Mobile/Outbound' is no longer available. Reset to 'Any'." |
| No data after filtering | `st.warning()` | "No data available for Email/Outbound. Try selecting 'Any' or adjusting global filters." |
| No channels available | `st.warning()` | "No channel data available with current global filters." |

### Consistency with Existing Patterns

- Use same selectbox styling as Stage selector
- Follow existing `da_streamlit_utils` patterns (similar to `stage_selectbox`)
- Persist state across navigation (like Stage Granularity does)
- Use `on_change` callbacks for state management (existing pattern)

## Testing & Verification

### Manual Testing Checklist

**Basic Functionality:**
- [ ] "Any" selected → shows all data (baseline behavior)
- [ ] Select specific Channel/Direction → charts update to show only that data
- [ ] Switch between different channels → data updates correctly each time
- [ ] Navigate away and back → selection persists

**Integration with Global Filters:**
- [ ] Set global Channel filter to "Web OR Email"
- [ ] Verify Optionality selector only shows Web and Email options (+ Any)
- [ ] Select "Any" → shows Web + Email data (not all channels)
- [ ] Clear global filters → Optionality selector shows all channels again

**Edge Cases:**
- [ ] Load v1 (Explainability Extract) data → channel selector shows channels without Direction
- [ ] Apply global filter that excludes all channels → selector shows only "Any" with warning
- [ ] Select channel, then apply global filter that excludes it → resets to "Any" with info message
- [ ] Select channel with very few interactions → charts render correctly (not empty error)

**All Four Visualizations:**
- [ ] "Optionality" chart (propensity vs optionality) updates correctly
- [ ] "Optionality Funnel" chart updates correctly (or shows warning if v1 data)
- [ ] "Optionality Trend" chart updates correctly
- [ ] "Offer Variation" chart updates correctly

**Performance:**
- [ ] Large dataset (50k+ interactions) → filtering is responsive
- [ ] Switching between channels doesn't cause long delays
- [ ] No memory leaks from repeated filtering

### Automated Testing

Add unit tests in `tests/test_decision_analyzer.py`:

```python
def test_filtered_sample_with_channel_filter():
    """Test filtered_sample property applies channel filter correctly."""
    # Create mock data with Channel and Direction
    # Set page_channel_expr in mock session_state
    # Assert filtered_sample returns correct subset

def test_filtered_sample_without_filters():
    """Test filtered_sample falls back to sample when no filters set."""
    # Create mock data
    # Don't set any page filters
    # Assert filtered_sample == sample

def test_get_available_channel_directions_with_direction():
    """Test channel/direction extraction for v2 data."""
    # Create mock v2 data with Channel and Direction
    # Call get_available_channel_directions()
    # Assert returns ["Channel/Direction"] format

def test_get_available_channel_directions_without_direction():
    """Test channel-only extraction for v1 data."""
    # Create mock v1 data with Channel only (no Direction)
    # Call get_available_channel_directions()
    # Assert returns ["Channel"] format

def test_get_available_channel_directions_no_channel():
    """Test graceful handling when no channel columns exist."""
    # Create mock data without Channel or Direction
    # Call get_available_channel_directions()
    # Assert returns empty list
```

### UI Testing

Perform manual UI testing with:
- Sample v2 data (Decision Analyzer export)
- Sample v1 data (Explainability Extract)
- Edge case: dataset with only 1 channel
- Edge case: dataset with 10+ channels

Take screenshots of:
- Default state (Any selected)
- Specific channel selected
- Warning messages (no data, invalid selection)

### Documentation Updates

**Files to update:**

1. **`python/docs/source/GettingStartedWithDecisionAnalysis.rst`**
   - Add section describing channel filter feature
   - Include screenshot showing the sidebar selector
   - Document the "Any" vs specific channel behavior

2. **Inline help text**
   - Already included in widget `help` parameter

3. **CLAUDE.md** (if needed)
   - Add any conventions for page-level filters

## Future Extensibility

### Adding Filter to Other Pages

To add the channel filter to another page (e.g., Win/Loss Analysis):

```python
# In the page file:
with st.session_state["sidebar"]:
    channel_direction_selector()  # Import from da_streamlit_utils

# Use filtered_sample instead of sample:
filtered_data = st.session_state.decision_data.filtered_sample
```

### Adding Other Page-Level Filters

The pattern established here can be reused for other filters:

```python
# Example: Issue filter
def issue_selector():
    """Render issue selector in sidebar."""
    # Similar pattern to channel_direction_selector()
    # Store in st.session_state.page_issue_filter
    # Store expression in st.session_state.page_issue_expr

# In DecisionAnalyzer.filtered_sample:
@property
def filtered_sample(self):
    result = self.sample

    # Apply channel filter
    if (channel_expr := st.session_state.get("page_channel_expr")):
        result = apply_filter(result, [channel_expr])

    # Apply issue filter
    if (issue_expr := st.session_state.get("page_issue_expr")):
        result = apply_filter(result, [issue_expr])

    return result
```

### Multiple Page Filters

If we want to combine multiple page filters:

```python
@property
def filtered_sample(self):
    """Apply all page-level filters."""
    filters = []

    # Collect all active page filters
    if (expr := st.session_state.get("page_channel_expr")):
        filters.append(expr)
    if (expr := st.session_state.get("page_issue_expr")):
        filters.append(expr)
    # ... more filters

    if filters:
        return apply_filter(self.sample, filters)
    return self.sample
```

## Open Questions & Decisions

### Resolved

1. **Q: Should the channel filter override global filters?**
   - **A:** No, it should respect global filters (intersection approach)

2. **Q: Should we use Channel only or Channel/Direction?**
   - **A:** Channel/Direction combinations (when Direction exists)

3. **Q: Where should the filter apply?**
   - **A:** After global filters and sampling, before visualizations

4. **Q: Should we use @cached_property?**
   - **A:** No, use regular @property due to session_state dependency

### Open (for future consideration)

1. **Should we add a visual indicator in the page title when filtered?**
   - Current design: No indicator
   - Alternative: Add `# Optionality Analysis · 📊 Web/Inbound`
   - Decision: Can add in future if user feedback suggests it's needed

2. **Should we add a "Reset all page filters" button?**
   - Current design: Manual reset to "Any"
   - Alternative: Global reset button in sidebar
   - Decision: Add when we have multiple page filters

3. **Should we cache filtered_sample at page level?**
   - Current design: No caching, compute on each access
   - Alternative: Cache in session_state per page
   - Decision: Monitor performance, add caching if needed

## Implementation Timeline

1. **Phase 1: Core Implementation** (Main development)
   - Add `filtered_sample` property to DecisionAnalyzer
   - Add helper functions to `da_streamlit_utils`
   - Update Optionality Analysis page
   - Manual testing with sample data

2. **Phase 2: Edge Case Handling** (Refinement)
   - Test with v1 data (no Direction column)
   - Test with global filters applied
   - Test invalid selection scenarios
   - Polish user feedback messages

3. **Phase 3: Testing & Documentation** (Quality assurance)
   - Add unit tests
   - Comprehensive UI testing
   - Update documentation
   - Take screenshots for docs

4. **Phase 4: Review & Merge** (Finalization)
   - Code review
   - Address feedback
   - Merge to master

## Success Criteria

This feature is considered successful when:

1. ✅ Users can select a specific Channel/Direction from the sidebar dropdown
2. ✅ All four Optionality analyses update to show only that channel's data
3. ✅ "Any" option shows aggregated data across all channels (respecting global filters)
4. ✅ Selection persists when navigating away and returning
5. ✅ Edge cases are handled gracefully with user-friendly messages
6. ✅ Works with both v1 and v2 data formats
7. ✅ Performance is acceptable on large datasets
8. ✅ Code passes all tests and follows project conventions
9. ✅ Documentation is updated

## References

- **Related Pages:** Global Data Filters page (existing global filter pattern)
- **Related Code:** `stage_selectbox()` in `da_streamlit_utils.py` (similar UI pattern)
- **Related Functionality:** `apply_filter()` utility function
- **Data Schema:** `column_schema.py` (Channel and Direction definitions)

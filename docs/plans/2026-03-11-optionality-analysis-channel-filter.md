# Optionality Analysis Channel Filter Implementation Plan

**Status:** Implemented and tested
**Date completed:** 2026-03-11

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Channel/Direction filter dropdown to the Optionality Analysis page sidebar that filters all visualizations to a specific channel combination.

**Architecture:** Implement a `filtered_sample` property on DecisionAnalyzer that applies page-level filters stored in Streamlit session_state. Add helper functions for channel selection UI that respect global filters. Update Optionality Analysis page to use filtered data.

**Tech Stack:** Python, Polars, Streamlit, pytest

---

## Prerequisites

- Branch: `feature/optionality-channel-filter` (create from current branch)
- Design doc: `docs/plans/2026-03-11-optionality-analysis-channel-filter-design.md`
- Sample data for testing: `data/sample_eev2.parquet` (v2 with Channel/Direction)

---

## Task 1: Add filtered_sample Property to DecisionAnalyzer

**Files:**
- Modify: `python/pdstools/decision_analyzer/DecisionAnalyzer.py`
- Create: `python/tests/test_decision_analyzer_filtered_sample.py`

**Step 1: Create test file with basic fixtures**

Create `python/tests/test_decision_analyzer_filtered_sample.py`:

```python
"""Tests for DecisionAnalyzer.filtered_sample property."""
import polars as pl
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_data_v2():
    """Create sample v2 data with Channel and Direction."""
    return pl.DataFrame({
        "Interaction ID": ["I1", "I1", "I2", "I2", "I3", "I3"],
        "Channel": ["Web", "Web", "Email", "Email", "Mobile", "Mobile"],
        "Direction": ["Inbound", "Inbound", "Outbound", "Outbound", "Inbound", "Inbound"],
        "Action": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "Decision Time": [
            "2024-01-01", "2024-01-01", "2024-01-02",
            "2024-01-02", "2024-01-03", "2024-01-03"
        ],
        "Priority": [100.0, 90.0, 85.0, 80.0, 75.0, 70.0],
    }).with_columns(pl.col("Decision Time").str.strptime(pl.Datetime))


@pytest.fixture
def mock_decision_analyzer(sample_data_v2):
    """Create a mock DecisionAnalyzer with sample property."""
    from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer

    # Create analyzer with minimal init
    analyzer = DecisionAnalyzer.__new__(DecisionAnalyzer)

    # Mock the sample property to return our test data
    type(analyzer).sample = property(lambda self: sample_data_v2.lazy())

    return analyzer
```

**Step 2: Write failing test for filtered_sample without filters**

Add to `python/tests/test_decision_analyzer_filtered_sample.py`:

```python
def test_filtered_sample_without_filters(mock_decision_analyzer, sample_data_v2):
    """When no page filters set, filtered_sample should return sample unchanged."""
    with patch('streamlit.session_state', new_callable=MagicMock) as mock_st:
        mock_st.get.return_value = None  # No page_channel_expr

        result = mock_decision_analyzer.filtered_sample
        expected = sample_data_v2.lazy()

        # Compare collected DataFrames
        assert result.collect().equals(expected.collect())
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py::test_filtered_sample_without_filters -v`

Expected: FAIL with "AttributeError: 'DecisionAnalyzer' object has no attribute 'filtered_sample'"

**Step 4: Write failing test for filtered_sample with channel filter**

Add to `python/tests/test_decision_analyzer_filtered_sample.py`:

```python
def test_filtered_sample_with_channel_filter(mock_decision_analyzer, sample_data_v2):
    """When page_channel_expr is set, filtered_sample should apply the filter."""
    with patch('streamlit.session_state', new_callable=MagicMock) as mock_st:
        # Set up a channel filter for Web/Inbound
        channel_expr = (pl.col("Channel") == "Web") & (pl.col("Direction") == "Inbound")
        mock_st.get.return_value = channel_expr

        result = mock_decision_analyzer.filtered_sample

        # Should only have Web/Inbound rows (first 2 rows)
        result_collected = result.collect()
        assert len(result_collected) == 2
        assert (result_collected["Channel"] == "Web").all()
        assert (result_collected["Direction"] == "Inbound").all()
```

**Step 5: Run test to verify it fails**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py::test_filtered_sample_with_channel_filter -v`

Expected: FAIL with "AttributeError: 'DecisionAnalyzer' object has no attribute 'filtered_sample'"

**Step 6: Write failing test for non-Streamlit context**

Add to `python/tests/test_decision_analyzer_filtered_sample.py`:

```python
def test_filtered_sample_outside_streamlit(mock_decision_analyzer, sample_data_v2):
    """When not in Streamlit context, filtered_sample should return sample."""
    # Don't mock streamlit - let ImportError occur naturally
    result = mock_decision_analyzer.filtered_sample
    expected = sample_data_v2.lazy()

    assert result.collect().equals(expected.collect())
```

**Step 7: Run test to verify it fails**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py::test_filtered_sample_outside_streamlit -v`

Expected: FAIL with "AttributeError: 'DecisionAnalyzer' object has no attribute 'filtered_sample'"

**Step 8: Implement filtered_sample property**

In `python/pdstools/decision_analyzer/DecisionAnalyzer.py`, add after the `sample` property (around line 680):

```python
@property
def filtered_sample(self):
    """Sample data with page-level filters applied.

    Reads filter expressions from st.session_state.page_channel_expr if available.
    Falls back to unfiltered sample if no page filters are set or not in Streamlit context.

    This property is not cached because it depends on mutable session_state.
    Page-level code should cache the result locally if needed for performance.

    Returns
    -------
    pl.LazyFrame
        Sampled data with page filters applied, or unfiltered sample if no filters.

    Examples
    --------
    >>> # In Streamlit app with channel filter set
    >>> st.session_state.page_channel_expr = pl.col("Channel") == "Web"
    >>> filtered = analyzer.filtered_sample  # Only Web channel data
    >>>
    >>> # Without filters
    >>> filtered = analyzer.filtered_sample  # Same as .sample
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

**Step 9: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py -v`

Expected: All 3 tests PASS

**Step 10: Commit**

```bash
git add python/pdstools/decision_analyzer/DecisionAnalyzer.py python/tests/test_decision_analyzer_filtered_sample.py
git commit -m "feat(decision_analyzer): add filtered_sample property for page-level filters

Add filtered_sample property that applies page-level filters from
session_state on top of the sample data. Falls back to unfiltered
sample when no filters are set or outside Streamlit context.

This enables page-specific filtering (e.g., channel selection) without
modifying the core sample or affecting other pages."
```

---

## Task 2: Add Channel Selection Helper Functions

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`
- Modify: `python/tests/test_decision_analyzer_filtered_sample.py`

**Step 1: Write failing test for get_available_channel_directions with v2 data**

Add to `python/tests/test_decision_analyzer_filtered_sample.py`:

```python
def test_get_available_channel_directions_v2(sample_data_v2):
    """Should return sorted Channel/Direction combinations for v2 data."""
    from pdstools.app.decision_analyzer.da_streamlit_utils import (
        get_available_channel_directions,
    )

    result = get_available_channel_directions(sample_data_v2.lazy())

    expected = [
        "Email/Outbound",
        "Mobile/Inbound",
        "Web/Inbound",
    ]
    assert result == expected
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py::test_get_available_channel_directions_v2 -v`

Expected: FAIL with "ImportError: cannot import name 'get_available_channel_directions'"

**Step 3: Write failing test for get_available_channel_directions with v1 data**

Add to `python/tests/test_decision_analyzer_filtered_sample.py`:

```python
@pytest.fixture
def sample_data_v1():
    """Create sample v1 data with Channel only (no Direction)."""
    return pl.DataFrame({
        "Interaction ID": ["I1", "I1", "I2", "I2"],
        "Channel": ["Web", "Web", "Email", "Email"],
        "Action": ["A1", "A2", "A3", "A4"],
        "Decision Time": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "Priority": [100.0, 90.0, 85.0, 80.0],
    }).with_columns(pl.col("Decision Time").str.strptime(pl.Datetime))


def test_get_available_channel_directions_v1(sample_data_v1):
    """Should return sorted Channel values for v1 data without Direction."""
    from pdstools.app.decision_analyzer.da_streamlit_utils import (
        get_available_channel_directions,
    )

    result = get_available_channel_directions(sample_data_v1.lazy())

    expected = ["Email", "Web"]
    assert result == expected
```

**Step 4: Run test to verify it fails**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py::test_get_available_channel_directions_v1 -v`

Expected: FAIL with "ImportError: cannot import name 'get_available_channel_directions'"

**Step 5: Write failing test for get_available_channel_directions with no channel data**

Add to `python/tests/test_decision_analyzer_filtered_sample.py`:

```python
@pytest.fixture
def sample_data_no_channel():
    """Create sample data without Channel or Direction columns."""
    return pl.DataFrame({
        "Interaction ID": ["I1", "I2"],
        "Action": ["A1", "A2"],
        "Decision Time": ["2024-01-01", "2024-01-02"],
    }).with_columns(pl.col("Decision Time").str.strptime(pl.Datetime))


def test_get_available_channel_directions_no_channel(sample_data_no_channel):
    """Should return empty list when no Channel column exists."""
    from pdstools.app.decision_analyzer.da_streamlit_utils import (
        get_available_channel_directions,
    )

    result = get_available_channel_directions(sample_data_no_channel.lazy())

    assert result == []
```

**Step 6: Run test to verify it fails**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py::test_get_available_channel_directions_no_channel -v`

Expected: FAIL with "ImportError: cannot import name 'get_available_channel_directions'"

**Step 7: Implement get_available_channel_directions**

In `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`, add at the end of the file (before any cache functions):

```python
def get_available_channel_directions(sample_df: pl.LazyFrame) -> list[str]:
    """Get list of Channel/Direction combinations from the sample data.

    Handles both v2 data (with Direction) and v1 data (Channel only).
    Respects any filters already applied to the sample (e.g., global filters).

    Parameters
    ----------
    sample_df : pl.LazyFrame
        The sample data (after global filters applied).

    Returns
    -------
    list[str]
        Sorted list of "Channel/Direction" strings for v2 data,
        or sorted list of "Channel" strings for v1 data,
        or empty list if no channel data available.

    Examples
    --------
    >>> # V2 data with Direction
    >>> get_available_channel_directions(sample)
    ['Email/Outbound', 'Web/Inbound', 'Web/Outbound']

    >>> # V1 data without Direction
    >>> get_available_channel_directions(sample)
    ['Email', 'Mobile', 'Web']
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
        channels = (
            sample_df
            .select("Channel")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        return sorted(channels)
    else:
        # No channel data at all
        return []
```

**Step 8: Run tests to verify they pass**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py -v`

Expected: All tests PASS (including previous filtered_sample tests)

**Step 9: Commit**

```bash
git add python/pdstools/app/decision_analyzer/da_streamlit_utils.py python/tests/test_decision_analyzer_filtered_sample.py
git commit -m "feat(decision_analyzer): add get_available_channel_directions helper

Add helper function to extract unique Channel/Direction combinations
from sample data. Handles v2 (with Direction), v1 (Channel only), and
missing channel data gracefully.

Used for populating the channel filter dropdown with available options."
```

**Step 10: Implement channel_direction_selector UI function**

Add to `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`, after `get_available_channel_directions`:

```python
def _update_channel_filter():
    """Callback to update filter expression when channel selection changes.

    Converts the UI value (e.g., "Web/Inbound") into a Polars filter expression
    and stores it in session_state.page_channel_expr for use by filtered_sample.
    """
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


def channel_direction_selector():
    """Render channel/direction selector in sidebar.

    Stores selection in st.session_state.page_channel_filter (UI value)
    and st.session_state.page_channel_expr (Polars filter expression).

    Handles edge cases:
    - Resets to "Any" if previous selection no longer available
    - Shows warning if no channel data available
    - Updates filter expression when selection changes

    The filter is applied via DecisionAnalyzer.filtered_sample property.

    Examples
    --------
    >>> # In Streamlit sidebar
    >>> with st.sidebar:
    >>>     channel_direction_selector()
    >>>
    >>> # Use filtered data in analysis
    >>> filtered_data = st.session_state.decision_data.filtered_sample
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
    st.selectbox(
        "Channel / Direction",
        options=options,
        index=options.index(current),
        key="_channel_direction_widget",
        on_change=_update_channel_filter,
        help="Filter analysis to a specific channel. 'Any' shows all channels that pass global filters.",
    )
```

**Step 11: Commit**

```bash
git add python/pdstools/app/decision_analyzer/da_streamlit_utils.py
git commit -m "feat(decision_analyzer): add channel_direction_selector UI component

Add Streamlit sidebar widget for channel/direction selection. Handles
edge cases like invalid selections and missing channel data. Updates
session_state with both UI value and Polars filter expression.

No unit tests for Streamlit UI components (tested manually)."
```

---

## Task 3: Update Optionality Analysis Page

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`

**Step 1: Add channel selector to sidebar**

In `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`, modify the sidebar section (lines 21-23):

```python
ensure_data()
st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()  # NEW: Add channel filter
```

Update the import at the top (line 4):

```python
from da_streamlit_utils import (
    channel_direction_selector,  # NEW
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)
```

**Step 2: Add filtered data computation and empty check**

After the sidebar section (around line 24), add:

```python
# Apply channel filter to sample data
filtered_data = st.session_state.decision_data.filtered_sample

# Check for empty results when a specific channel is selected
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.count()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()
```

**Step 3: Update Optionality section to use filtered_data**

In the first container (lines 25-50), change line 42:

```python
# OLD
df=st.session_state.decision_data.sample,

# NEW
df=filtered_data,
```

**Step 4: Update Optionality Funnel section to use filtered_data**

In the Optionality Funnel container (lines 52-61), change line 58:

```python
# OLD
df=st.session_state.decision_data.sample

# NEW
df=filtered_data
```

**Step 5: Update Optionality Trend section to use filtered_data**

In the Optionality Trend container (lines 63-86), change line 72:

```python
# OLD
st.session_state.decision_data.get_optionality_data_with_trend(df=st.session_state.decision_data.sample)

# NEW
st.session_state.decision_data.get_optionality_data_with_trend(df=filtered_data)
```

**Step 6: Note about Offer Variation section**

The Offer Variation section (lines 88-104) currently hardcodes `stage="Output"` and doesn't accept a `df` parameter. This is by design - it always shows the final output stage. No changes needed here.

Add a comment above line 97 to document this:

```python
# Note: Offer Variation uses Output stage and doesn't accept df parameter.
# It's intentionally excluded from channel filtering to show global variation.
st.plotly_chart(
```

**Step 7: Verify complete file**

The updated file should look like this at the top:

```python
# python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py
import polars as pl
import streamlit as st
from da_streamlit_utils import (
    channel_direction_selector,
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)

# TODO cosmetics nicer color scheme for the stages - do consistently in all plots then
# TODO for optionality plot allow to overlay propensity but also maybe add priority?
# TODO think more about the "offer variation" or personalization index - I now calculated PI from the AUC of the variation curve

"# Optionality Analysis"

"""
Analysis of the number of actions per customer. Do we have enough options for people? Global filters can
be applied like everywhere to look at e.g. a certain group of issues.
"""
ensure_data()
st.session_state["sidebar"] = st.sidebar
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()

# Apply channel filter to sample data
filtered_data = st.session_state.decision_data.filtered_sample

# Check for empty results when a specific channel is selected
if st.session_state.get("page_channel_filter", "Any") != "Any":
    filtered_count = filtered_data.select(pl.count()).collect().item()
    if filtered_count == 0:
        st.warning(
            f"No data available for {st.session_state.page_channel_filter}. "
            "Try selecting 'Any' or adjusting global filters."
        )
        st.stop()

with st.container(border=True):
    "## Optionality"

    st.caption(
        "Showing the number of actions available and the average of the highest "
        "propensity when there are this number of actions. Generally, with more actions "
        "you would expect higher propensities as there is more to choose from."
    )

    stage_selectbox(
        key="optionality_stage",
        default="Arbitration",
    )

    st.plotly_chart(
        st.session_state.decision_data.plot.propensity_vs_optionality(
            stage=st.session_state.get("optionality_stage", "Arbitration"),
            df=filtered_data,
        ),
        width="stretch",
    )

    st.caption(
        "Propensity is only available from the Arbitration stage onward, "
        "as it is calculated only for the actions that AI prioritizes at that point."
    )

if st.session_state.decision_data.extract_type != "explainability_extract":
    with st.container(border=True):
        "## Optionality Funnel"
        "Distribution of Available action by Stage"
        if st.session_state.decision_data.extract_type == "decision_analyzer":
            st.plotly_chart(
                st.session_state.decision_data.plot.optionality_funnel(df=filtered_data),
                width="stretch",
            )


with st.container(border=True):
    "## Optionality Trend"

    st.caption(
        "Showing the number of unique actions over time - so you can spot significant "
        "changes in the number of available actions."
    )

    optionality_data_with_trend_per_stage = (
        st.session_state.decision_data.get_optionality_data_with_trend(df=filtered_data)
        .group_by(["day", st.session_state.decision_data.level])
        .agg(nOffers=pl.col("nOffers").max())
        .sort("day")
    )

    fig, warning = st.session_state.decision_data.plot.optionality_trend(
        optionality_data_with_trend_per_stage,
    )
    if warning is not None:
        st.warning(warning)
    st.plotly_chart(
        fig,
        width="stretch",
    )

with st.container(border=True):
    "## Offer Variation"

    st.caption(
        "How much variation is there in the offers? Does everyone get the same few actions or "
        "is there a lot of variation in what we are offering?"
    )

    # Note: Offer Variation uses Output stage and doesn't accept df parameter.
    # It's intentionally excluded from channel filtering to show global variation.
    st.plotly_chart(
        st.session_state.decision_data.plot.action_variation(stage="Output"),
        width="stretch",
    )
    action_variability_stats = st.session_state.decision_data.get_offer_variability_stats("Output")
    st.caption(
        f"{action_variability_stats['n90']} actions win in 90% of the final decisions made. "
        f"The personalization index is **{round(action_variability_stats['gini'], 3)}**."
    )
```

**Step 8: Manual smoke test**

Run: `uv run pdstools decision_analyzer --data-path data/sample_eev2.parquet`

Test checklist:
- [ ] Page loads without errors
- [ ] Channel/Direction selector appears in sidebar with "Any" + channel options
- [ ] Selecting "Any" shows all data (no error)
- [ ] Selecting a specific channel updates all 3 filtered charts (Optionality, Funnel, Trend)
- [ ] Offer Variation section still shows all data (not filtered)
- [ ] No console errors in browser

Expected: All checks pass

**Step 9: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py
git commit -m "feat(optionality): add channel filter to Optionality Analysis page

Add channel/direction selector to sidebar and update all relevant
visualizations to use filtered_sample. Offer Variation section
intentionally uses unfiltered data to show global variation.

Includes empty data check with user-friendly warning message."
```

---

## Task 4: Integration Testing

**Files:**
- Test with: `data/sample_eev2.parquet`

**Step 1: Test basic functionality**

Run: `uv run pdstools decision_analyzer --data-path data/sample_eev2.parquet`

Navigate to Optionality Analysis page.

Test checklist:
- [ ] Default state: "Any" selected, all data shown
- [ ] Select "Web/Inbound": charts update to show only Web/Inbound data
- [ ] Select "Email/Outbound": charts update to show only Email/Outbound data
- [ ] Switch back to "Any": charts show all data again
- [ ] Navigate to another page and back: selection persists

**Step 2: Test with global filters**

1. Go to Global Data Filters page
2. Filter by Channel: select only "Web" and "Email"
3. Navigate to Optionality Analysis

Test checklist:
- [ ] Channel selector shows only: "Any", channels containing "Web" or "Email"
- [ ] Selecting "Any" shows Web + Email data (not all channels)
- [ ] Selecting specific channel works correctly
- [ ] Clear global filters
- [ ] Optionality selector shows all channels again

**Step 3: Test edge cases**

Test 1: Invalid selection after global filter
1. Select "Mobile/Inbound" in Optionality page
2. Go to Global Data Filters and exclude Mobile
3. Return to Optionality Analysis
- [ ] Should show info message "Previous selection 'Mobile/Inbound' is no longer available"
- [ ] Should reset to "Any"
- [ ] Charts should display without error

Test 2: Empty results
1. Select a channel with very few interactions
2. Add restrictive global filters
- [ ] If result is empty, should show warning message
- [ ] Page should stop rendering charts (st.stop())

**Step 4: Test all visualizations update correctly**

With "Web/Inbound" selected:
- [ ] Optionality chart: shows only Web/Inbound propensity vs optionality
- [ ] Optionality Funnel: shows only Web/Inbound distribution by stage
- [ ] Optionality Trend: shows only Web/Inbound unique actions over time
- [ ] Offer Variation: shows ALL channels (not filtered) ← Important!

**Step 5: Test performance**

With large dataset (50k+ interactions):
- [ ] Switching channels is responsive (< 2 seconds)
- [ ] No memory leaks (check browser dev tools)
- [ ] No excessive recomputation

**Step 6: Document any issues**

If any issues found:
1. Create GitHub issue with reproduction steps
2. Note in this plan under "Known Issues" section
3. Fix if critical, otherwise defer

**Step 7: Run full test suite**

Run: `uv run pytest python/tests/test_decision_analyzer_filtered_sample.py -v`

Expected: All tests PASS

**Step 8: Check for regressions**

Run broader test suite:
Run: `uv run pytest python/tests/ -k decision_analyzer -v`

Expected: No new failures (existing skipped tests are OK)

---

## Task 5: Documentation

**Files:**
- Modify: `python/docs/source/GettingStartedWithDecisionAnalysis.rst`

**Step 1: Add channel filter section to documentation**

In `python/docs/source/GettingStartedWithDecisionAnalysis.rst`, find the "Optionality Analysis" section and add a subsection about the channel filter:

```rst
Optionality Analysis
^^^^^^^^^^^^^^^^^^^^

The Optionality Analysis page helps you understand whether customers have enough
action choices at each stage of the decision pipeline.

**Channel Filtering**

You can filter the Optionality Analysis to a specific channel/direction combination
using the dropdown in the sidebar:

- **Any** (default): Shows aggregated data across all channels that pass global filters
- **Specific channel**: Shows data only for that Channel/Direction combination
  (e.g., "Web/Inbound", "Email/Outbound")

The filter applies to:
- Optionality chart (propensity vs number of actions)
- Optionality Funnel (action distribution by stage)
- Optionality Trend (unique actions over time)

The Offer Variation section is intentionally not filtered, as it shows the global
variation in action selection across all channels.

**Note:** The channel filter respects global filters from the Global Data Filters
page. Only channels that exist after global filters are applied will appear in
the dropdown.

.. image:: images/optionality_channel_filter.png
   :alt: Channel filter dropdown in Optionality Analysis
   :align: center
```

**Step 2: Build docs to verify**

Run: `cd python/docs && make html`

Expected: Build succeeds with no warnings about the new section

**Step 3: Commit**

```bash
git add python/docs/source/GettingStartedWithDecisionAnalysis.rst
git commit -m "docs: document channel filter in Optionality Analysis

Add documentation for the new channel/direction filter feature in
the Optionality Analysis page. Explains filter behavior, scope,
and interaction with global filters."
```

---

## Task 6: Final Cleanup and PR Preparation

**Step 1: Review all changes**

Run: `git log --oneline feature/optionality-channel-filter --not master`

Expected: See all commits from this implementation:
- feat(decision_analyzer): add filtered_sample property
- feat(decision_analyzer): add get_available_channel_directions helper
- feat(decision_analyzer): add channel_direction_selector UI component
- feat(optionality): add channel filter to Optionality Analysis page
- docs: document channel filter in Optionality Analysis

**Step 2: Run pre-commit checks**

Run: `git diff master --name-only | xargs pre-commit run --files`

Expected: All checks pass (ruff, trailing whitespace, etc.)

**Step 3: Verify no unintended changes**

Run: `git diff master --stat`

Expected changes:
- `python/pdstools/decision_analyzer/DecisionAnalyzer.py`: ~15 lines added
- `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`: ~80 lines added
- `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`: ~15 lines added/modified
- `python/tests/test_decision_analyzer_filtered_sample.py`: ~100 lines added (new file)
- `python/docs/source/GettingStartedWithDecisionAnalysis.rst`: ~25 lines added

Total: ~235 lines added/modified

**Step 4: Final integration test**

Run complete smoke test:
1. Start app: `uv run pdstools decision_analyzer --data-path data/sample_eev2.parquet`
2. Navigate to Optionality Analysis
3. Test all functionality from Task 4
4. Verify no console errors

Expected: Everything works as designed

**Step 5: Update implementation plan status**

Mark this plan as complete by adding at the top:

```markdown
**Status:** ✅ Implemented and tested
**Date completed:** YYYY-MM-DD
```

---

## Success Criteria

Implementation is complete when:

- [x] `filtered_sample` property added to DecisionAnalyzer with tests
- [x] Channel selection helper functions implemented with tests
- [x] Optionality Analysis page updated to use filtered data
- [x] All visualizations correctly filter (except Offer Variation)
- [x] Edge cases handled gracefully with user-friendly messages
- [x] Works with both v2 (Channel/Direction) and v1 (Channel only) data
- [x] Integration tests pass all checklist items
- [x] Documentation updated
- [x] Pre-commit checks pass
- [x] No regressions in existing tests

---

## Known Issues

*Document any issues discovered during implementation here*

None at this time.

---

## Future Enhancements

Consider for future iterations:

1. **Visual indicator in page title** when channel is filtered (e.g., "Optionality Analysis · 📊 Web/Inbound")
2. **Add channel filter to other pages** (Win/Loss Analysis, Action Distribution)
3. **Multiple page filters** (channel + issue + stage) with combined filtering
4. **Reset all page filters button** in sidebar
5. **Performance optimization** if filtering becomes slow on very large datasets

---

## Notes

- Offer Variation section intentionally not filtered to show global variation
- Page-level filters stored in session_state but only used by pages that explicitly call filtered_sample
- Pattern is extensible to other page-level filters (issue, stage, etc.)
- No @cached_property used due to session_state dependency (see design doc)

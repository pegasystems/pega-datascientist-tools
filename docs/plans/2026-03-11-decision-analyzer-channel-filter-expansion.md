# Decision Analyzer Channel Filter Expansion Implementation Plan

**Status:** Implemented and tested
**Date completed:** 2026-03-11

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend channel/direction filter to 6 additional Decision Analyzer pages (Action Distribution, Action Funnel, Global Sensitivity, Win/Loss Analysis, Offer Quality Analysis, Thresholding Analysis, Arbitration Component Distribution).

**Architecture:** Apply the proven pattern from Optionality Analysis (PR #573) to each page: add `channel_direction_selector()` to sidebar, compute `filtered_data` from `filtered_sample` property, replace `sample` references with `filtered_data`.

**Tech Stack:** Python, Streamlit, Polars, Playwright (for UI tests)

---

## Prerequisites

**Branch:** `feature/optionality-channel-filter` (PR #573)
**Design doc:** `docs/plans/2026-03-11-decision-analyzer-channel-filter-expansion-design.md`

---

## Task 1: Switch to Feature Branch

**Step 1: Checkout the feature branch**

Run: `git checkout feature/optionality-channel-filter`

Expected: "Switched to branch 'feature/optionality-channel-filter'"

**Step 2: Pull latest changes**

Run: `git pull origin feature/optionality-channel-filter`

Expected: "Already up to date" or new commits pulled

**Step 3: Verify infrastructure exists**

Run: `grep -n "def filtered_sample" python/pdstools/decision_analyzer/DecisionAnalyzer.py`

Expected: Should find the `filtered_sample` property (added in PR #573)

Run: `grep -n "def channel_direction_selector" python/pdstools/app/decision_analyzer/da_streamlit_utils.py`

Expected: Should find the selector function

---

## Task 2: Add Channel Filter to Action Distribution

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py`

**Step 1: Read current file**

Run: `cat python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py | head -40`

Verify current imports and sidebar structure.

**Step 2: Update imports**

In `python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py`, modify the import section (lines 4-9):

```python
from da_streamlit_utils import (
    channel_direction_selector,  # ADD THIS LINE
    get_current_index,
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)
```

**Step 3: Add channel selector to sidebar**

After line 37 (inside the sidebar context), add:

```python
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()  # ADD THIS LINE
```

**Step 4: Add filtered data computation**

After the sidebar section (around line 42), add:

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

**Step 5: Find and replace sample references**

Search for uses of `st.session_state.decision_data.sample` in the file:

Run: `grep -n "decision_data.sample" python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py`

Replace each occurrence with `filtered_data`. The main plot calls should be updated to use `df=filtered_data` if they accept a `df` parameter.

**Step 6: Manual smoke test**

Run: `uv run pdstools decision_analyzer`

Navigate to Action Distribution page.

Test checklist:
- [ ] Channel selector appears in sidebar
- [ ] Selecting "Any" shows data without errors
- [ ] Selecting specific channel updates the treemap
- [ ] No errors in terminal or browser console

**Step 7: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py
git commit -m "feat(decision_analyzer): add channel filter to Action Distribution

Add channel/direction selector to sidebar and update visualizations
to use filtered_sample. Follows pattern established in Optionality Analysis.

Part of multi-page channel filter rollout."
```

---

## Task 3: Add Channel Filter to Action Funnel

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py`

**Step 1: Update imports**

In `python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py`, modify imports (lines 6-12):

```python
from da_streamlit_utils import (
    channel_direction_selector,  # ADD THIS LINE
    ensure_data,
    ensure_funnel,
    get_current_index,
    polars_lazyframe_hashing,
    stage_level_selector,
)
```

**Step 2: Add channel selector to sidebar**

After line 48 (inside sidebar), add:

```python
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()  # ADD THIS LINE
```

**Step 3: Add filtered data computation**

After sidebar section (around line 60), add:

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

**Step 4: Update decision_funnel function to accept channel filter**

Modify the cached function (lines 37-43):

```python
@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def decision_funnel(
    scope,
    level=None,
    return_df=False,
    channel_filter=None,  # ADD THIS PARAMETER
):
    # Note: channel_filter added to cache key, but we use filtered_data from outside
    return st.session_state.decision_data.plot.decision_funnel(scope=scope, return_df=return_df)
```

**Step 5: Update funnel call to use filtered data**

The decision_funnel method likely needs to be updated to accept a `df` parameter. Check if the method signature supports it:

Run: `grep -A 10 "def decision_funnel" python/pdstools/decision_analyzer/plots.py`

If it accepts `df` parameter, update the call (around line 71):

```python
remanining_funnel, filtered_funnel = decision_funnel(
    scope=st.session_state.scope,
    level=st.session_state.decision_data.level,
    channel_filter=st.session_state.get("page_channel_filter", "Any"),  # ADD FOR CACHE KEY
)
```

And pass the filtered data to the DecisionAnalyzer method if possible.

**Step 6: Important - Do NOT filter the filter impact table**

The filter impact table (lines 94+) uses raw `decision_data.decision_data` - this should remain unfiltered as it shows all filter events. Add a comment to clarify:

```python
# Note: Filter impact table uses raw decision_data to show all filter events,
# not affected by channel filter (by design)
data = (
    st.session_state.decision_data.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
    ...
)
```

**Step 7: Manual smoke test**

Run: `uv run pdstools decision_analyzer`

Navigate to Action Funnel page.

Test checklist:
- [ ] Channel selector appears
- [ ] Funnel updates when channel selected
- [ ] Filter impact table still shows all data (not filtered)

**Step 8: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py
git commit -m "feat(decision_analyzer): add channel filter to Action Funnel

Add channel selector to sidebar and update funnel visualization to use
filtered_sample. Filter impact table intentionally remains unfiltered
to show all filter events across channels."
```

---

## Task 4: Add Channel Filter to Global Sensitivity

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py`

**Step 1: Update imports**

In `python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py`, modify imports (line 2):

```python
from da_streamlit_utils import channel_direction_selector, ensure_data, get_current_index
```

**Step 2: Add channel selector to sidebar**

After line 22 (inside sidebar), add:

```python
with st.session_state["sidebar"]:
    channel_direction_selector()  # ADD THIS LINE

    st.number_input(
        "Top-N actions that define Winning",
        ...
    )
```

**Step 3: Add filtered data computation**

After sidebar section (around line 30), add:

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

**Step 4: Update sensitivity plot call**

Check if `plot.sensitivity()` accepts a `df` parameter:

Run: `grep -A 5 "def sensitivity" python/pdstools/decision_analyzer/plots.py`

If it does, update the call (around line 35):

```python
st.plotly_chart(
    st.session_state.decision_data.plot.sensitivity(
        st.session_state.win_rank,
        df=filtered_data,  # ADD IF SUPPORTED
    ),
    width="stretch",
)
```

**Step 5: Update global_winloss_distribution plot call**

Similarly, check and update the call (around line 55):

```python
st.plotly_chart(
    st.session_state.decision_data.plot.global_winloss_distribution(
        level=st.session_state.glob_sensitivity_scope,
        win_rank=st.session_state.win_rank,
        df=filtered_data,  # ADD IF SUPPORTED
    ),
    width="stretch",
)
```

**Step 6: Manual smoke test**

Run: `uv run pdstools decision_analyzer`

Navigate to Global Sensitivity page.

Test checklist:
- [ ] Channel selector appears
- [ ] Sensitivity chart updates when channel selected
- [ ] Win/Loss distribution updates when channel selected

**Step 7: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py
git commit -m "feat(decision_analyzer): add channel filter to Global Sensitivity

Add channel selector to sidebar and update sensitivity and win/loss
visualizations to use filtered_sample."
```

---

## Task 5: Add Channel Filter to Win/Loss Analysis

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/7_Win_Loss_Analysis.py`

**Step 1: Update imports**

In `python/pdstools/app/decision_analyzer/pages/7_Win_Loss_Analysis.py`, modify imports (lines 2-7):

```python
from da_streamlit_utils import (
    channel_direction_selector,  # ADD THIS LINE
    ensure_data,
    get_current_index,
    get_data_filters,
    show_filtered_counts,
)
```

**Step 2: Add channel selector to sidebar**

After line 40 (inside sidebar, before other controls):

```python
with st.session_state["sidebar"]:
    channel_direction_selector()  # ADD THIS LINE AT TOP

    scope_options = st.session_state.decision_data.getPossibleScopeValues()
    ...
```

**Step 3: Add filtered data computation**

After sidebar section (around line 63), add:

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

**Step 4: Replace sample references**

Find all uses of `decision_data.sample`:

Run: `grep -n "decision_data.sample" python/pdstools/app/decision_analyzer/pages/7_Win_Loss_Analysis.py`

Replace each with `filtered_data`. Key locations:
- Line 68: `get_data_filters(filtered_data, ...)`
- Line 77: `get_first_level_stats(filtered_data)`
- Line 80: `get_first_level_stats(filtered_data, ...)`
- Any plot method calls

**Step 5: Manual smoke test**

Run: `uv run pdstools decision_analyzer`

Navigate to Win/Loss Analysis page.

Test checklist:
- [ ] Channel selector appears
- [ ] Comparison group filter still works
- [ ] Win/loss charts update when channel selected

**Step 6: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/7_Win_Loss_Analysis.py
git commit -m "feat(decision_analyzer): add channel filter to Win/Loss Analysis

Add channel selector to sidebar and update all visualizations and local
filters to use filtered_sample. Enables channel-specific competitive analysis."
```

---

## Task 6: Add Channel Filter to Offer Quality Analysis

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/9_Offer_Quality_Analysis.py`

**Step 1: Update imports**

In `python/pdstools/app/decision_analyzer/pages/9_Offer_Quality_Analysis.py`, modify imports (lines 5-9):

```python
from da_streamlit_utils import (
    channel_direction_selector,  # ADD THIS LINE
    ensure_data,
    stage_level_selector,
    stage_selectbox,
)
```

**Step 2: Add channel selector to sidebar**

After line 57 (inside sidebar, after stage_level_selector):

```python
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()  # ADD THIS LINE

    propensityTH = (
        st.slider(
            ...
        )
    )
```

**Step 3: Add filtered data computation**

After sidebar section (around line 80), add:

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

**Step 4: Update threshold calculations to use filtered data**

The threshold calculations (lines 45-46) use `getThresholdingData()`. Check if this method respects the sample or uses raw data:

Run: `grep -A 10 "def getThresholdingData" python/pdstools/decision_analyzer/DecisionAnalyzer.py`

If it uses sample internally, the filtered_sample property will handle it automatically. If not, we may need to pass `df=filtered_data`.

**Step 5: Update plot calls to use filtered data**

Find offer quality plot calls and update them to use `df=filtered_data` if they accept the parameter.

**Step 6: Manual smoke test**

Run: `uv run pdstools decision_analyzer`

Navigate to Offer Quality Analysis page.

Test checklist:
- [ ] Channel selector appears
- [ ] Thresholds update correctly
- [ ] Quality pie charts update when channel selected
- [ ] Trend charts update when channel selected

**Step 7: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/9_Offer_Quality_Analysis.py
git commit -m "feat(decision_analyzer): add channel filter to Offer Quality Analysis

Add channel selector to sidebar and update quality metrics and visualizations
to use filtered_sample. Threshold calculations respect channel filter."
```

---

## Task 7: Add Channel Filter to Thresholding Analysis

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py`

**Step 1: Update imports**

In `python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py`, modify imports (line 6):

```python
from da_streamlit_utils import channel_direction_selector, ensure_data
```

**Step 2: Add channel selector to sidebar**

After line 43 (inside sidebar, before threshold sliders):

```python
with st.sidebar:
    channel_direction_selector()  # ADD THIS LINE AT TOP

    prop_range = da.getThresholdingData("Propensity", quantile_range=[0, 100])["Threshold"].to_list()
    ...
```

**Step 3: Add filtered data computation**

After the `da = st.session_state.decision_data` line (around line 23), add:

```python
da = st.session_state.decision_data

# Apply channel filter to sample data
filtered_data = da.filtered_sample

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

**Step 4: Check pre-aggregation usage**

The file uses `getPreaggregatedFilterView` (line 29). Verify this respects the sample:

Run: `grep -A 10 "def getPreaggregatedFilterView" python/pdstools/decision_analyzer/DecisionAnalyzer.py`

If it uses `self.sample` internally, the filtered_sample will be automatically respected. If it uses raw `decision_data`, we need to filter explicitly.

**Step 5: Update arb_data to use filtered data**

If `getPreaggregatedFilterView` doesn't respect sample, modify line 28-36:

```python
# Use filtered sample for threshold analysis
arb_data = (
    filtered_data  # CHANGED from getPreaggregatedFilterView
    .filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
    .select(
        pl.col("Propensity").explode(),
        pl.col("Priority").explode(),
        pl.col("Decisions"),
    )
    .collect()
)
```

**Step 6: Manual smoke test**

Run: `uv run pdstools decision_analyzer`

Navigate to Thresholding Analysis page.

Test checklist:
- [ ] Channel selector appears
- [ ] Threshold sliders work correctly
- [ ] Charts update when channel selected
- [ ] Threshold impact calculations are correct

**Step 7: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py
git commit -m "feat(decision_analyzer): add channel filter to Thresholding Analysis

Add channel selector to sidebar and update threshold analysis to use
filtered_sample. Threshold impact calculations respect channel filter."
```

---

## Task 8: Add Channel Filter to Arbitration Component Distribution

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py`

**Step 1: Update imports**

In `python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py`, modify imports (lines 3-10):

```python
from da_streamlit_utils import (
    channel_direction_selector,  # ADD THIS LINE
    ensure_data,
    get_current_index,
    stage_level_selector,
    stage_selectbox,
    st_component_overview,
    st_priority_component_distribution,
)
```

**Step 2: Determine available components using filtered data**

The current code (line 37) uses `decision_data.sample` to determine available columns. Update to use filtered data after we compute it.

**Step 3: Add channel selector to sidebar**

After line 47 (inside sidebar, after stage_level_selector):

```python
with st.session_state["sidebar"]:
    stage_level_selector()
    channel_direction_selector()  # ADD THIS LINE

    stage_selectbox(default="Arbitration")
    ...
```

**Step 4: Add filtered data computation**

After sidebar section (around line 58), add:

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

**Step 5: Move component detection after filtered data**

Move lines 36-38 to after the filtered_data computation:

```python
# Determine which components are actually present in the filtered data
available_cols = set(filtered_data.collect_schema().names())  # CHANGED to use filtered_data
component_options = [c for c in PRIO_COMPONENTS if c in available_cols]
```

**Step 6: Check if component methods accept df parameter**

Run: `grep -A 5 "def all_components_distribution" python/pdstools/decision_analyzer/DecisionAnalyzer.py`

If the method accepts `df` parameter, update the call (line 70):

```python
overview_data = st.session_state.decision_data.all_components_distribution(
    st.session_state.scope,
    stage=st.session_state.stage,
    df=filtered_data,  # ADD IF SUPPORTED
)
```

**Step 7: Update other component distribution calls**

Search for other method calls that use sample data and update them to use `df=filtered_data` if supported.

**Step 8: Manual smoke test**

Run: `uv run pdstools decision_analyzer`

Navigate to Arbitration Component Distribution page.

Test checklist:
- [ ] Channel selector appears
- [ ] Component overview updates when channel selected
- [ ] Violin plots update correctly
- [ ] ECDF plots update correctly

**Step 9: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py
git commit -m "feat(decision_analyzer): add channel filter to Arbitration Component Distribution

Add channel selector to sidebar and update all component distribution
visualizations to use filtered_sample."
```

---

## Task 9: Add Playwright UI Tests

**Files:**
- Create: `python/tests/test_channel_filter_ui.py`

**Step 1: Check if Playwright is installed**

Run: `uv pip list | grep playwright`

If not installed:

Run: `uv pip install playwright pytest-playwright`
Run: `playwright install`

**Step 2: Create Playwright test file**

Create `python/tests/test_channel_filter_ui.py`:

```python
"""Playwright tests for channel filter across Decision Analyzer pages."""
import pytest
from playwright.sync_api import Page, expect


PAGES_WITH_FILTER = [
    "Action_Distribution",
    "Action_Funnel",
    "Global_Sensitivity",
    "Win_Loss_Analysis",
    "Optionality_Analysis",
    "Offer_Quality_Analysis",
    "Thresholding_Analysis",
    "Arbitration_Component_Distribution",
]


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the Streamlit app.

    Assumes app is running on localhost:8501.
    Start app before running tests: uv run pdstools decision_analyzer
    """
    return "http://localhost:8501"


def test_channel_selector_present_on_all_pages(page: Page, app_url: str):
    """Verify channel selector appears on all expected pages."""
    for page_name in PAGES_WITH_FILTER:
        page.goto(f"{app_url}/pages/{page_name}")
        page.wait_for_load_state("networkidle")

        # Check if selector is visible
        selector = page.locator('label:has-text("Channel / Direction")')
        expect(selector).to_be_visible(timeout=5000)


def test_channel_filter_updates_visualization(page: Page, app_url: str):
    """Test that selecting a channel updates the visualization."""
    page.goto(f"{app_url}/pages/Action_Distribution")
    page.wait_for_load_state("networkidle")

    # Wait for initial render
    page.wait_for_selector('label:has-text("Channel / Direction")', timeout=10000)

    # Select a specific channel (assumes Web/Inbound exists in test data)
    channel_select = page.locator('label:has-text("Channel / Direction")').locator('..').locator('select')
    channel_select.select_option("Web/Inbound")

    # Wait for update (spinner should appear and disappear)
    page.wait_for_load_state("networkidle", timeout=10000)

    # Verify no error messages
    error = page.locator('[data-testid="stException"]')
    expect(error).not_to_be_visible()


def test_channel_filter_persists_across_navigation(page: Page, app_url: str):
    """Test that channel selection persists when navigating between pages."""
    # Select channel on Action Distribution
    page.goto(f"{app_url}/pages/Action_Distribution")
    page.wait_for_load_state("networkidle")

    channel_select = page.locator('label:has-text("Channel / Direction")').locator('..').locator('select')
    channel_select.select_option("Email/Outbound")
    page.wait_for_load_state("networkidle")

    # Navigate to Win Loss Analysis
    page.goto(f"{app_url}/pages/Win_Loss_Analysis")
    page.wait_for_load_state("networkidle")

    # Verify selection persisted
    channel_select_new = page.locator('label:has-text("Channel / Direction")').locator('..').locator('select')
    expect(channel_select_new).to_have_value("Email/Outbound")


def test_overview_page_has_no_channel_filter(page: Page, app_url: str):
    """Verify Overview page does NOT have channel filter."""
    page.goto(f"{app_url}/pages/Overview")
    page.wait_for_load_state("networkidle")

    # Channel selector should not be present
    selector = page.locator('label:has-text("Channel / Direction")')
    expect(selector).not_to_be_visible()


def test_any_option_resets_filter(page: Page, app_url: str):
    """Test that selecting 'Any' shows all data."""
    page.goto(f"{app_url}/pages/Action_Distribution")
    page.wait_for_load_state("networkidle")

    # Select specific channel
    channel_select = page.locator('label:has-text("Channel / Direction")').locator('..').locator('select')
    channel_select.select_option("Web/Inbound")
    page.wait_for_load_state("networkidle")

    # Select 'Any'
    channel_select.select_option("Any")
    page.wait_for_load_state("networkidle")

    # Should not show any errors
    error = page.locator('[data-testid="stException"]')
    expect(error).not_to_be_visible()
```

**Step 3: Create pytest configuration for Playwright**

If not already configured, add to `pytest.ini` or `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "ui: UI tests requiring running Streamlit app (deselect with '-m \"not ui\"')",
]
```

**Step 4: Mark tests as UI tests**

Add marker to each test function:

```python
@pytest.mark.ui
def test_channel_selector_present_on_all_pages(page: Page, app_url: str):
    ...
```

**Step 5: Create test documentation**

Create `python/tests/README_UI_TESTS.md`:

```markdown
# UI Tests for Decision Analyzer

## Prerequisites

1. Install Playwright:
   ```bash
   uv pip install playwright pytest-playwright
   playwright install
   ```

2. Start the Decision Analyzer app:
   ```bash
   uv run pdstools decision_analyzer --data-path examples/data/sample.parquet
   ```

## Running Tests

Run all UI tests:
```bash
uv run pytest python/tests/test_channel_filter_ui.py -v
```

Run specific test:
```bash
uv run pytest python/tests/test_channel_filter_ui.py::test_channel_selector_present_on_all_pages -v
```

Skip UI tests in regular test runs:
```bash
uv run pytest python/tests/ -m "not ui" -v
```

## Test Coverage

- Channel selector presence on all 8 pages
- Visualization updates when channel selected
- Filter persistence across page navigation
- Overview page exclusion
- "Any" option behavior
```

**Step 6: Run Playwright tests**

Start the app in one terminal:
```bash
uv run pdstools decision_analyzer
```

In another terminal, run tests:
```bash
uv run pytest python/tests/test_channel_filter_ui.py -v
```

Expected: All tests PASS

**Step 7: Commit**

```bash
git add python/tests/test_channel_filter_ui.py python/tests/README_UI_TESTS.md
git commit -m "test: add Playwright UI tests for channel filter

Add automated UI tests to verify channel filter behavior across all
Decision Analyzer pages. Tests cover selector presence, visualization
updates, persistence, and edge cases."
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `python/docs/source/GettingStartedWithDecisionAnalyzer.rst`

**Step 1: Read current documentation**

Run: `cat python/docs/source/GettingStartedWithDecisionAnalyzer.rst | grep -A 20 "Optionality"`

Find appropriate location to add channel filtering section.

**Step 2: Add channel filtering section**

Add a new section after the page descriptions (suggested location: after the "Pages Overview" section):

```rst
Channel Filtering
^^^^^^^^^^^^^^^^^

Many Decision Analyzer pages support filtering to a specific channel/direction combination
using the **Channel / Direction** dropdown in the sidebar.

**Available Pages**

The channel filter is available on these analysis pages:

* Action Distribution
* Action Funnel
* Global Sensitivity
* Win/Loss Analysis
* Optionality Analysis
* Offer Quality Analysis
* Thresholding Analysis
* Arbitration Component Distribution

.. note::
   The Overview page intentionally shows global metrics across all channels and does
   not have a channel filter.

**How it works**

- **Any** (default): Shows aggregated data across all channels that pass global filters
- **Specific channel**: Shows data only for that Channel/Direction combination
  (e.g., "Web/Inbound", "Email/Outbound")

The channel filter appears at the top of the sidebar on each page. Your selection persists
as you navigate between pages, allowing you to maintain the same channel focus across
different analyses.

**Interaction with Global Filters**

The channel filter respects global filters from the Global Data Filters page. Only
channels that exist after global filters are applied will appear in the dropdown.

If you select a channel and then apply a global filter that excludes that channel, the
filter will automatically reset to "Any" with an informational message.

**Example Workflow**

1. Navigate to Action Distribution page
2. Select "Web/Inbound" from Channel / Direction dropdown
3. Analyze Web/Inbound specific patterns in the treemap and bar charts
4. Navigate to Win/Loss Analysis page
5. Same "Web/Inbound" filter is active, showing competitive dynamics for this channel
6. Compare results across different channels by switching the filter

This enables channel-specific analysis workflows while maintaining consistency across pages.
```

**Step 3: Build documentation to verify**

Run: `cd python/docs && make html`

Expected: Build succeeds with no warnings

**Step 4: View rendered documentation**

Run: `open python/docs/build/html/GettingStartedWithDecisionAnalyzer.html`

Verify the new section renders correctly.

**Step 5: Commit**

```bash
git add python/docs/source/GettingStartedWithDecisionAnalyzer.rst
git commit -m "docs: document channel filter across all Decision Analyzer pages

Add comprehensive documentation for the channel filter feature, including
available pages, behavior, interaction with global filters, and example
workflow. Updates getting started guide with channel filtering section."
```

---

## Task 11: Final Testing and PR Update

**Step 1: Run full test suite**

Run: `uv run pytest python/tests/ -k decision_analyzer -v`

Expected: All existing tests pass (skipped tests unchanged)

**Step 2: Comprehensive manual test across all pages**

Run: `uv run pdstools decision_analyzer`

Test checklist for each of the 7 updated pages:

**Action Distribution:**
- [ ] Channel selector visible
- [ ] "Any" shows all data
- [ ] "Web/Inbound" updates treemap correctly
- [ ] Navigate away and back, selection persists

**Action Funnel:**
- [ ] Channel selector visible
- [ ] Funnel updates with channel selected
- [ ] Filter impact table shows all data (not filtered)

**Global Sensitivity:**
- [ ] Channel selector visible
- [ ] Sensitivity chart updates
- [ ] Win/loss distribution updates

**Win/Loss Analysis:**
- [ ] Channel selector visible
- [ ] Comparison group still works
- [ ] Win/loss matrices update

**Offer Quality Analysis:**
- [ ] Channel selector visible
- [ ] Quality metrics update
- [ ] Pie charts update

**Thresholding Analysis:**
- [ ] Channel selector visible
- [ ] Threshold sliders work
- [ ] Impact charts update

**Arbitration Component Distribution:**
- [ ] Channel selector visible
- [ ] Component overview updates
- [ ] Violin and ECDF plots update

**Cross-page consistency:**
- [ ] Filter persists when navigating between any two filtered pages
- [ ] Overview page has NO channel filter
- [ ] Warning messages are consistent

**Step 3: Test edge cases**

**Edge case 1: Empty result**
1. Select a channel with few interactions
2. Apply restrictive global filter
3. Verify warning appears if result is empty

**Edge case 2: Invalid selection**
1. Select "Mobile/Inbound"
2. Go to Global Data Filters and exclude Mobile
3. Return to any filtered page
4. Verify info message about reset to "Any"

**Step 4: Run Playwright tests**

Start app:
```bash
uv run pdstools decision_analyzer
```

Run UI tests:
```bash
uv run pytest python/tests/test_channel_filter_ui.py -v
```

Expected: All tests PASS

**Step 5: Update PR description**

Update PR #573 on GitHub with expanded scope:

**New PR description:**

```markdown
## Channel Filter for Decision Analyzer Pages

Add channel/direction filtering to 8 Decision Analyzer pages, enabling channel-specific analysis workflows.

### Pages Updated

- [x] Optionality Analysis (original scope)
- [x] Action Distribution
- [x] Action Funnel
- [x] Global Sensitivity
- [x] Win/Loss Analysis
- [x] Offer Quality Analysis
- [x] Thresholding Analysis
- [x] Arbitration Component Distribution

### Features

- **Channel/Direction selector** in sidebar with "Any" + available channels
- **Filter persistence** across page navigation
- **Respects global filters** from Global Data Filters page
- **Edge case handling** with user-friendly messages
- **Consistent UX** across all pages

### Infrastructure

- `DecisionAnalyzer.filtered_sample` property for page-level filtering
- `get_available_channel_directions()` helper for channel extraction
- `channel_direction_selector()` UI component with edge case handling

### Testing

- ✅ Playwright UI tests for all pages
- ✅ Manual testing checklist completed
- ✅ Edge cases verified (empty results, invalid selections)
- ✅ No regressions in existing tests

### Documentation

- ✅ Getting Started guide updated with channel filtering section
- ✅ Design doc: `docs/plans/2026-03-11-decision-analyzer-channel-filter-expansion-design.md`
- ✅ Implementation plan: `docs/plans/2026-03-11-decision-analyzer-channel-filter-expansion.md`

### Exclusions

- Overview page intentionally remains global (no channel filter)
- Global Data Filters page not affected
```

**Step 6: Review commit history**

Run: `git log --oneline feature/optionality-channel-filter --not origin/master | head -20`

Expected commits:
- feat(decision_analyzer): add channel filter to Action Distribution
- feat(decision_analyzer): add channel filter to Action Funnel
- feat(decision_analyzer): add channel filter to Global Sensitivity
- feat(decision_analyzer): add channel filter to Win/Loss Analysis
- feat(decision_analyzer): add channel filter to Offer Quality Analysis
- feat(decision_analyzer): add channel filter to Thresholding Analysis
- feat(decision_analyzer): add channel filter to Arbitration Component Distribution
- test: add Playwright UI tests for channel filter
- docs: document channel filter across all Decision Analyzer pages

**Step 7: Push to remote**

Run: `git push origin feature/optionality-channel-filter`

Expected: Push succeeds, PR #573 updates with new commits

---

## Success Criteria

Implementation is complete when:

- [x] Channel selector added to all 7 target pages
- [x] All visualizations respect channel filter
- [x] Filter state persists across navigation
- [x] Overview page remains global (no filter)
- [x] Edge cases handled with appropriate messages
- [x] Playwright tests written and passing
- [x] Manual testing checklist completed for all pages
- [x] Documentation updated
- [x] No regressions in existing tests
- [x] PR #573 updated with expanded scope
- [x] All commits follow convention and reference design doc

---

## Troubleshooting

### Issue: Plot methods don't accept `df` parameter

**Symptom:** `TypeError: plot_method() got an unexpected keyword argument 'df'`

**Solution:** Check method signature in `plots.py`. If it doesn't accept `df`, you may need to:
1. Update the method to accept `df` parameter (preferred)
2. Or pass filtered_data through DecisionAnalyzer by temporarily setting it (not recommended)

**Example fix:**
```python
# In plots.py
def sensitivity(self, win_rank, df=None):  # ADD df parameter
    data = df if df is not None else self.sample  # Use df if provided
    # ... rest of method
```

### Issue: Pre-aggregated data not respecting filter

**Symptom:** Thresholding Analysis shows all data even with channel filter

**Solution:** Check if `getPreaggregatedFilterView` uses `self.sample`. If it uses raw data:

```python
# Use filtered_data directly instead of pre-aggregation
arb_data = (
    filtered_data
    .filter(pl.col(da.level).is_in(da.stages_from_arbitration_down))
    # ... rest of query
)
```

### Issue: Playwright tests fail with timeout

**Symptom:** Tests fail with "Timeout 30000ms exceeded"

**Solution:**
1. Verify app is running: `curl http://localhost:8501`
2. Increase timeout: `expect(selector).to_be_visible(timeout=60000)`
3. Check page URL is correct (may need `/pages/` prefix)
4. Wait for network idle before checks: `page.wait_for_load_state("networkidle")`

### Issue: Empty data warning appears unexpectedly

**Symptom:** Warning shows "No data available" when there should be data

**Solution:**
1. Check if global filters are too restrictive
2. Verify `filtered_sample` returns data: `print(filtered_data.select(pl.count()).collect())`
3. Check if channel exists in data: `print(get_available_channel_directions(sample))`

---

## Notes

- Each page follows the exact same pattern from Optionality Analysis
- Infrastructure already exists from PR #573, no core library changes needed
- Filter impact table in Action Funnel intentionally not filtered (shows all events)
- Playwright tests require app to be running manually (not automated start/stop)
- Documentation assumes standard sidebar layout with Stage Granularity selector

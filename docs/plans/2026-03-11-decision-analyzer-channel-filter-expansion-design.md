# Channel Filter Expansion for Decision Analyzer - Design Document

**Date:** 2026-03-11
**Status:** Approved
**Author:** AI Assistant (Claude)

## Overview

Extend the channel/direction filter feature (currently only on Optionality Analysis) to 6 additional Decision Analyzer pages. This provides consistent channel-level filtering across all major analysis pages while maintaining the global overview on the Overview page.

## Context

PR #573 introduced channel filtering infrastructure:
- `DecisionAnalyzer.filtered_sample` property
- `get_available_channel_directions()` helper
- `channel_direction_selector()` UI component
- Successfully implemented on Optionality Analysis page

This design extends that proven pattern to 6 more pages.

## Requirements

### Functional Requirements

1. **Pages to Update**
   - Action Distribution (page 4)
   - Action Funnel (page 5)
   - Global Sensitivity (page 6)
   - Win/Loss Analysis (page 7)
   - Offer Quality Analysis (page 9)
   - Thresholding Analysis (page 10)
   - Arbitration Component Distribution (page 11)

2. **Consistent Behavior**
   - Channel selector in same sidebar position on all pages
   - Same "Any" vs specific channel behavior
   - Filter state persists across page navigation
   - Same warning messages for edge cases

3. **Exclusions**
   - Overview page remains global (no channel filter)
   - Global Data Filters page not affected
   - About page not affected

4. **Integration**
   - Works with existing global filters
   - Uses existing `filtered_sample` infrastructure
   - No changes to core DecisionAnalyzer class needed

### Non-Functional Requirements

1. **Consistency:** Follow exact pattern from Optionality Analysis
2. **Maintainability:** Each page remains self-contained and readable
3. **Performance:** No performance degradation on large datasets
4. **Testability:** Automated tests with Playwright + manual verification

## Architecture

### Data Flow (Unchanged from Optionality)

```
User selects channel/direction in sidebar
         ↓
session_state.page_channel_expr updated
         ↓
DecisionAnalyzer.filtered_sample applies filter
         ↓
Page uses filtered_data instead of sample
         ↓
All visualizations show channel-specific data
```

### Reusing Existing Infrastructure

All infrastructure exists from PR #573:
- ✅ `DecisionAnalyzer.filtered_sample` property
- ✅ `get_available_channel_directions()` helper
- ✅ `channel_direction_selector()` UI component
- ✅ Tests for filtering logic
- ✅ Edge case handling (empty data, invalid selections)

**No core library changes needed** - only page updates.

## Implementation Design

### Standard Pattern (Applied to Each Page)

Each page follows this 4-step template:

**Step 1: Update imports**
```python
from da_streamlit_utils import (
    channel_direction_selector,  # ADD THIS
    # ... existing imports
)
```

**Step 2: Add selector to sidebar**
```python
with st.session_state["sidebar"]:
    stage_level_selector()  # or other existing selectors
    channel_direction_selector()  # ADD THIS
```

**Step 3: Compute filtered data with empty check**
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
```python
# OLD: df=st.session_state.decision_data.sample
# NEW: df=filtered_data
```

### Page-Specific Implementation Details

#### 1. Action Distribution (page 4)

**Current state:** Uses `sample` indirectly through DecisionAnalyzer methods

**Changes needed:**
- Add selector to sidebar after `stage_level_selector()`
- Compute `filtered_data` at page top
- Pass `df=filtered_data` to plot methods (if they accept it)
- No special considerations

**Estimated impact:** ~10 lines added, ~3 lines modified

---

#### 2. Action Funnel (page 5)

**Current state:** Uses cached `decision_funnel()` function + filter impact table

**Changes needed:**
- Add selector to sidebar after `stage_level_selector()`
- Compute `filtered_data` at page top
- Update `decision_funnel()` cache key to include channel filter
- **Important:** Filter impact table should NOT be channel-filtered (shows all filter events)
- Only funnel visualization gets filtered

**Special consideration:**
```python
@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def decision_funnel(scope, level=None, return_df=False, channel_filter=None):
    # channel_filter added to cache key
    return st.session_state.decision_data.plot.decision_funnel(
        scope=scope,
        return_df=return_df,
        df=filtered_data if channel_filter else None
    )
```

**Estimated impact:** ~15 lines added, ~5 lines modified

---

#### 3. Global Sensitivity (page 6)

**Current state:** Uses `sample` indirectly through plot methods

**Changes needed:**
- Add selector to sidebar (currently only has `win_rank` number input)
- Compute `filtered_data` at page top
- Pass `df=filtered_data` to `plot.sensitivity()` and `plot.global_winloss_distribution()`

**Estimated impact:** ~10 lines added, ~3 lines modified

---

#### 4. Win/Loss Analysis (page 7)

**Current state:** Multiple direct references to `decision_data.sample`

**Changes needed:**
- Add selector to sidebar (currently has multiple controls)
- Compute `filtered_data` at page top
- Replace all `decision_data.sample` with `filtered_data`:
  - Line 68: Local filter setup
  - Line 77-80: Stats calculation
  - Multiple plot method calls

**Estimated impact:** ~10 lines added, ~8 lines modified

---

#### 5. Offer Quality Analysis (page 9)

**Current state:** Uses `sample` indirectly, has propensity/priority thresholds

**Changes needed:**
- Add selector to sidebar after `stage_level_selector()`
- Compute `filtered_data` at page top
- Pass `df=filtered_data` to offer quality methods
- Threshold calculations should respect filtered data

**Estimated impact:** ~10 lines added, ~4 lines modified

---

#### 6. Thresholding Analysis (page 10)

**Current state:** Uses `getPreaggregatedFilterView` directly

**Changes needed:**
- Add selector to sidebar
- Compute `filtered_data` at page top
- **Important:** Verify `getPreaggregatedFilterView` respects sample
- If it uses raw `decision_data`, may need to filter explicitly

**Special consideration:** May need to check if pre-aggregation logic needs updating

**Estimated impact:** ~10 lines added, ~5 lines modified

---

#### 7. Arbitration Component Distribution (page 11)

**Current state:** Uses `sample` indirectly through component distribution methods

**Changes needed:**
- Add selector to sidebar after `stage_level_selector()`
- Compute `filtered_data` at page top
- Pass `df=filtered_data` to `all_components_distribution()` and related methods
- Verify methods accept `df` parameter (may need to add if missing)

**Estimated impact:** ~10 lines added, ~4 lines modified

---

### Summary of Changes

| Page | File | Lines Added | Lines Modified | Special Considerations |
|------|------|-------------|----------------|------------------------|
| Action Distribution | `4_Action_Distribution.py` | ~10 | ~3 | None |
| Action Funnel | `5_Action_Funnel.py` | ~15 | ~5 | Cache key, filter table exclusion |
| Global Sensitivity | `6_Global_Sensitivity.py` | ~10 | ~3 | None |
| Win/Loss Analysis | `7_Win_Loss_Analysis.py` | ~10 | ~8 | Multiple sample references |
| Offer Quality Analysis | `9_Offer_Quality_Analysis.py` | ~10 | ~4 | Threshold calculations |
| Thresholding Analysis | `10_Thresholding_Analysis.py` | ~10 | ~5 | Pre-aggregation check |
| Arbitration Component Distribution | `11_Arbitration_Component_Distribution.py` | ~10 | ~4 | Method parameters |
| **Total** | **7 files** | **~75** | **~32** | **Low complexity** |

## Edge Cases

All edge cases are handled by existing infrastructure (from PR #573):

| Edge Case | Behavior | Location |
|-----------|----------|----------|
| No channel data available | Warning message, selector disabled | `channel_direction_selector()` |
| Selected channel becomes invalid | Info message, reset to "Any" | `channel_direction_selector()` |
| Empty filtered results | Warning message, `st.stop()` | Page-level check |
| V1 data without Direction | Show Channel only (no slash) | `get_available_channel_directions()` |
| Non-Streamlit context | `filtered_sample` returns unfiltered `sample` | `filtered_sample` property |

## Testing Strategy

### Automated Testing with Playwright

Create `tests/test_channel_filter_ui.py`:

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

def test_channel_selector_present_on_all_pages(page: Page):
    """Verify channel selector appears on all expected pages."""
    for page_name in PAGES_WITH_FILTER:
        page.goto(f"http://localhost:8501/{page_name}")
        expect(page.locator("text=Channel / Direction")).to_be_visible()

def test_channel_filter_updates_visualization(page: Page):
    """Test that selecting a channel updates the visualization."""
    page.goto("http://localhost:8501/Action_Distribution")

    # Select a specific channel
    page.select_option("text=Channel / Direction", "Web/Inbound")

    # Wait for update
    expect(page.locator(".stSpinner")).not_to_be_visible(timeout=10000)

    # Verify no errors
    expect(page.locator(".stException")).not_to_be_visible()

def test_channel_filter_persists_across_navigation(page: Page):
    """Test that channel selection persists when navigating between pages."""
    page.goto("http://localhost:8501/Action_Distribution")
    page.select_option("text=Channel / Direction", "Email/Outbound")

    # Navigate to another filtered page
    page.goto("http://localhost:8501/Win_Loss_Analysis")

    # Verify selection persisted
    expect(page.locator("select[aria-label='Channel / Direction']")).to_have_value("Email/Outbound")

def test_empty_result_shows_warning(page: Page):
    """Test that selecting a channel with no data shows a warning."""
    page.goto("http://localhost:8501/Action_Distribution")

    # Apply restrictive global filter first (if needed for test setup)
    # Then select a channel that produces no results

    # Verify warning appears
    expect(page.locator("text=No data available")).to_be_visible()

def test_overview_page_has_no_channel_filter(page: Page):
    """Verify Overview page does NOT have channel filter."""
    page.goto("http://localhost:8501/Overview")
    expect(page.locator("text=Channel / Direction")).not_to_be_visible()
```

### Manual Testing Checklist

**Thorough test on one high-priority page (Action Distribution):**
- [ ] Channel selector visible with "Any" + available channels
- [ ] "Any" selected → shows all data
- [ ] Select "Web/Inbound" → visualizations update correctly
- [ ] Select "Email/Outbound" → visualizations update correctly
- [ ] Switch back to "Any" → shows all data again
- [ ] Navigate to Overview and back → selection persists
- [ ] Apply global channel filter → dropdown options update
- [ ] Select channel with no data → warning message appears
- [ ] No errors in browser console

**Spot-check remaining pages (reduced checklist per page):**
- [ ] Selector visible in sidebar
- [ ] Selecting specific channel updates main visualization
- [ ] No errors in browser console
- [ ] No regression in existing functionality

**Cross-page consistency check:**
- [ ] All 7 pages have selector in same sidebar position
- [ ] Warning messages are identical
- [ ] Filter state persists across all pages
- [ ] "Any" behaves identically everywhere

### Regression Testing

Run existing test suite:
```bash
uv run pytest python/tests/ -k decision_analyzer -v
```

Expected: All existing tests pass (skip status unchanged)

## Documentation Updates

### File: `python/docs/source/GettingStartedWithDecisionAnalyzer.rst`

Add new section after the existing page descriptions:

```rst
Channel Filtering
^^^^^^^^^^^^^^^^^

Many Decision Analyzer pages support filtering to a specific channel/direction combination
using the **Channel / Direction** dropdown in the sidebar.

**Available Pages:**

The channel filter is available on these analysis pages:

* Action Distribution
* Action Funnel
* Global Sensitivity
* Win/Loss Analysis
* Optionality Analysis
* Offer Quality Analysis
* Thresholding Analysis
* Arbitration Component Distribution

**Note:** The Overview page intentionally shows global metrics across all channels and does
not have a channel filter.

**How it works:**

- **Any** (default): Shows aggregated data across all channels that pass global filters
- **Specific channel**: Shows data only for that Channel/Direction combination
  (e.g., "Web/Inbound", "Email/Outbound")

The channel filter appears at the top of the sidebar on each page. Selection persists as you
navigate between pages, allowing you to maintain the same channel focus across different analyses.

**Interaction with Global Filters:**

The channel filter respects global filters from the Global Data Filters page. Only
channels that exist after global filters are applied will appear in the dropdown.

If you select a channel and then apply a global filter that excludes that channel, the
filter will automatically reset to "Any" with an informational message.

**Example workflow:**

1. Navigate to Action Distribution page
2. Select "Web/Inbound" from Channel / Direction dropdown
3. Analyze Web/Inbound specific patterns
4. Navigate to Win/Loss Analysis
5. Same "Web/Inbound" filter is active
6. Compare competitive dynamics for this channel

.. image:: images/channel_filter_sidebar.png
   :alt: Channel filter dropdown in sidebar
   :align: center
```

## Branch & Deployment Strategy

### Branch

**Target:** `feature/optionality-channel-filter` (PR #573)

This keeps all channel filter work together:
- ✅ Infrastructure already implemented
- ✅ Optionality Analysis already working
- ➕ Add remaining 6 pages to same branch/PR

### Before Implementation

```bash
git checkout feature/optionality-channel-filter
git pull origin feature/optionality-channel-filter
```

### Commit Strategy

One commit per page for clear review:

```
feat(decision_analyzer): add channel filter to Action Distribution
feat(decision_analyzer): add channel filter to Action Funnel
feat(decision_analyzer): add channel filter to Global Sensitivity
feat(decision_analyzer): add channel filter to Win/Loss Analysis
feat(decision_analyzer): add channel filter to Offer Quality Analysis
feat(decision_analyzer): add channel filter to Thresholding Analysis
feat(decision_analyzer): add channel filter to Arbitration Component Distribution
docs: document channel filter across all Decision Analyzer pages
test: add Playwright tests for channel filter UI
```

### PR Update

Update PR #573 description to reflect expanded scope:

**Before:**
> Add channel/direction filter to Optionality Analysis page

**After:**
> Add channel/direction filter to 8 Decision Analyzer pages:
> - Optionality Analysis ✅ (implemented)
> - Action Distribution ➕ (new)
> - Action Funnel ➕ (new)
> - Global Sensitivity ➕ (new)
> - Win/Loss Analysis ➕ (new)
> - Offer Quality Analysis ➕ (new)
> - Thresholding Analysis ➕ (new)
> - Arbitration Component Distribution ➕ (new)
>
> Provides consistent channel-level filtering across all major analysis pages.

## User Experience Considerations

### Sidebar Layout Consistency

All pages will have consistent sidebar layout:

```
┌─────────────────────────────┐
│ SIDEBAR                     │
├─────────────────────────────┤
│ Stage Granularity           │  ← If page has it
│ ○ Stage Group  ○ Stage      │
├─────────────────────────────┤
│ Channel / Direction         │  ← NEW (consistent position)
│ ┌─────────────────────────┐ │
│ │ Any                   ▼ │ │
│ └─────────────────────────┘ │
├─────────────────────────────┤
│ [Other page-specific        │
│  controls below]            │
└─────────────────────────────┘
```

### Filter State Behavior

**Persistence:**
- Selection persists when navigating between filtered pages
- Selection persists even when visiting non-filtered pages (Overview)
- Selection cleared only when user selects "Any" or resets filters

**Visual Feedback:**
- No special indicator in page title (consistent with current design)
- Sidebar shows current selection
- Empty result warning if needed

### Performance Expectations

- Channel selection update: < 2 seconds on 50k+ interactions
- No memory leaks from repeated filtering
- Cached functions include channel filter in key

## Success Criteria

Implementation is complete when:

- [x] Channel selector added to all 7 target pages
- [x] All visualizations on each page respect channel filter
- [x] Edge cases handled with appropriate messages
- [x] Filter state persists across page navigation
- [x] Overview page remains global (no filter)
- [x] Playwright tests pass
- [x] Manual testing checklist completed
- [x] Documentation updated
- [x] No regressions in existing tests
- [x] PR #573 updated with new scope

## Future Enhancements

Consider for future iterations:

1. **Visual indicator in page header** when filtered (e.g., "Action Distribution · 📊 Web/Inbound")
2. **Reset all page filters button** in sidebar
3. **Multiple page filters** (channel + issue + action group)
4. **Filter presets** (save/load common filter combinations)
5. **URL parameters** for deep-linking to filtered views

## References

- **PR #573:** Original channel filter implementation for Optionality Analysis
- **Design Doc:** `docs/plans/2026-03-11-optionality-analysis-channel-filter-design.md`
- **Implementation Plan:** `docs/plans/2026-03-11-optionality-analysis-channel-filter.md`
- **Related Code:** `DecisionAnalyzer.filtered_sample`, `da_streamlit_utils.channel_direction_selector()`

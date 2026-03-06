# Offer Quality Visualization Improvements - Design

**Date:** 2026-03-06
**Status:** Approved

## Problem Statement

The Offer Quality Analysis page has two visualization issues:

1. **Limited stage visibility**: Pie charts are hardcoded to show only the first 5 stages. When users select "Stage" level (instead of "Stage Group"), they see only 5 of potentially 10+ stages. This limitation causes confusion, especially when the first 5 stages lack data variation (resulting in all-blue pie charts).

2. **Missing Overview integration**: The Overview page lacks an offer quality indicator, which is a key metric for understanding customer experience. Users must navigate to a separate page to see this important analysis.

## Solution Overview

Implement a dynamic grid layout for offer quality pie charts and add an Arbitration stage summary to the Overview page.

## Design Approach

**Selected Approach:** Dynamic Plotly subplots grid

- Remove the 5-stage hardcoded limit
- Create responsive grid layout (4 columns, dynamic rows)
- Add single-stage helper function for Overview integration
- Maintain all existing color schemes, interactivity, and legend behavior

**Alternatives considered:**
- Streamlit columns with individual charts: More complex, loses cohesive figure interactivity
- Hybrid with pagination: Adds unnecessary interaction burden for typical use cases

## Component Changes

### 1. Modified: `offer_quality_piecharts()` in `plots.py`

**Changes:**
- Remove hardcoded `[:5]` stage limit (lines 858-862)
- Calculate dynamic grid dimensions:
  ```python
  num_stages = len(AvailableNBADStages)
  num_cols = min(4, num_stages)
  num_rows = math.ceil(num_stages / num_cols)
  ```
- Set dynamic figure height: `400 + (num_rows - 1) * 300` pixels
- Add `vertical_spacing=0.10` for multi-row layouts
- Generate dynamic `specs` matrix for subplot grid
- Handle partial last rows (fill empty cells with `None`)

**Preserved:**
- Existing color scheme (green, blue, orange, red)
- Label ordering and mapping
- Shared legend behavior (only first chart shows legend)
- Data preparation and aggregation logic

### 2. New: `offer_quality_single_pie()` in `plots.py`

**Purpose:** Generate a single pie chart for a specific stage (used in Overview page).

**Parameters:**
- `df`: LazyFrame with offer quality data
- `stage`: Stage name to display (e.g., "Arbitration")
- `propensityTH`: Propensity threshold for relevance categorization
- `level`: Grouping level (Stage/Stage Group)

**Implementation:**
- Extract data for specified stage from aggregated dataframe
- Apply same label mapping and ordering as multi-chart view
- Return single pie chart with full legend
- Fixed height: 300px (compact for Overview context)
- Use identical color scheme for consistency

### 3. Modified: `9_Offer_Quality_Analysis.py`

**Changes:**
- Pass full `AvailableNBADStages` list to `offer_quality_piecharts()` (no slicing)

**Preserved:**
- All sidebar controls and threshold sliders
- Data preparation logic (`filtered_action_counts`, `get_offer_quality`)
- Trend chart section
- Warning messages for datasets without arbitration data

### 4. Modified: `3_Overview.py`

**New section in right column (below sensitivity chart):**

```python
"## :red[Offer Quality]"

if has_arbitration_data:
    """
    Shows the distribution of customer interactions by offer quality at the
    arbitration stage. Green indicates customers received relevant offers,
    while red shows customers without offers.
    """

    # Calculate offer quality with default thresholds
    action_counts = st.session_state.decision_data.filtered_action_counts(
        groupby_cols=["Stage Group", "Interaction ID"],
        priorityTH=<default_from_percentile>,
        propensityTH=0.05,
    )

    vf = st.session_state.decision_data.get_offer_quality(
        action_counts,
        group_by="Interaction ID"
    )

    st.plotly_chart(
        offer_quality_single_pie(
            vf,
            stage="Arbitration",
            propensityTH=0.05,
            level="Stage Group",
        ),
        use_container_width=True,
    )
else:
    st.warning(
        "No actions survive to the arbitration stage in this data set. "
        "Offer quality analysis is not available."
    )
```

**Key decisions:**
- Always show "Arbitration" stage (works for V1 and V2 data)
- Use default thresholds (5th percentile propensity, similar for priority)
- Reuse existing `has_arbitration_data` guard
- Place in right column for balanced layout

## Data Flow

### Offer Quality Analysis Page
1. User selects stage level (Stage Group or Stage) via sidebar
2. User adjusts propensity/priority thresholds via sliders
3. Compute `filtered_action_counts` with user thresholds
4. Compute `get_offer_quality` grouped by level and Interaction ID
5. Pass full stage list to `offer_quality_piecharts()`
6. Display dynamic grid with all stages

### Overview Page
1. Check `has_arbitration_data` at page load
2. If true, compute `filtered_action_counts` with default thresholds
3. Compute `get_offer_quality` grouped by Interaction ID only
4. Call `offer_quality_single_pie()` with stage="Arbitration"
5. Display in right column with descriptive text

## Grid Layout Calculations

### Multi-chart grid dimensions:
- **Columns**: `min(4, num_stages)` - maximum 4 columns
- **Rows**: `ceil(num_stages / num_cols)`
- **Figure height**: `400 + (num_rows - 1) * 300` pixels
  - Examples:
    - 5 stages → 2 rows → 700px
    - 10 stages → 3 rows → 1000px
    - 15 stages → 4 rows → 1300px

### Spacing:
- **Horizontal**: 0.15 (existing, works well for 4 columns)
- **Vertical**: 0.10 (new, provides breathing room between rows)

### Subplot specs:
```python
specs = [[{"type": "domain"}] * num_cols for _ in range(num_rows)]
# Handle partial last row by setting unused positions to None
```

### Single-chart sizing:
- **Height**: 300px (compact for Overview)
- **Legend**: Full legend displayed (not shared)

## Performance Considerations

- Plotly handles 20-30 pie charts efficiently in a single figure
- Scrollable container manages vertical height gracefully
- No pagination needed for typical datasets (5-15 stages)
- For extreme cases (50+ stages), browser scrolling handles the content
- Overview page impact minimal (single additional chart with cached data)

## Testing Strategy

1. **Visual regression**: Verify grid layout with 5, 10, 15, and 20+ stages
2. **Stage level switching**: Confirm all stages visible when switching between "Stage Group" and "Stage"
3. **Color consistency**: Verify color scheme matches between multi-chart and single-chart views
4. **Overview integration**: Test Overview page with V1 (explainability extract) and V2 (full pipeline) data
5. **Edge cases**:
   - Single stage dataset
   - Dataset without arbitration data
   - Empty stages (zero interactions)
6. **Responsive behavior**: Test with different browser widths

## V1 Data Compatibility

For V1 (explainability extract) data:
- The system artificially creates an "Arbitration" stage group (line 282-286 in DecisionAnalyzer.py)
- This ensures Overview page shows offer quality even for V1-only datasets
- Warning guards prevent showing quality analysis when no actions reach arbitration

## Success Criteria

1. Users can see all available stages in Offer Quality Analysis pie charts
2. Grid layout is readable and well-spaced for 2-20 stages
3. Overview page displays Arbitration stage offer quality for both V1 and V2 data
4. Color scheme and labels consistent across all views
5. No performance degradation with typical stage counts (up to 20)
6. Existing functionality (thresholds, trends) remains unchanged

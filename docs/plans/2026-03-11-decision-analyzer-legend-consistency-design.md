# Decision Analyzer Legend Consistency Design

**Date:** 2026-03-11
**Status:** Approved

## Problem Statement

Chart colors and legend ordering in the Decision Analysis Tool currently change whenever users interact with UI controls (channel selectors, dropdowns, filters). This happens because Plotly assigns colors dynamically based on the unique values present in the filtered data, which varies with each filter change.

**Example Issue:**
- User selects "Email" channel → Issues appear in one color scheme
- User switches to "SMS" channel → Same Issues now have different colors
- Result: Confusing for users trying to track specific dimensions across views

**Affected Dimensions:**
All categorical columns used for plot coloring, including:
- Core dimensions: `Issue`, `Group`, `Action`, `Treatment`
- Stage dimensions: `Stage`, `Stage Group`
- Channel dimensions: `Channel`, `Direction`
- Any other categorical columns used in plot color parameters

## Design Goals

1. **Consistency:** Legend colors remain stable throughout an analysis session
2. **Predictability:** Colors assigned based on all data values, not filtered subsets
3. **Performance:** Minimal impact on initial load time
4. **User Clarity:** Users understand why colors remain consistent despite filtering

## Solution: Lazy Initialization with Memoization

### Architecture

**Core Component: `DecisionAnalyzer.color_mappings` Property**

Add a `@cached_property` to the `DecisionAnalyzer` class that computes color assignments on first access:

```python
@cached_property
def color_mappings(self) -> dict[str, dict[str, str]]:
    """Compute consistent color mappings for all categorical dimensions.

    Returns a dictionary mapping dimension names to color dictionaries.
    Example: {
        "Issue": {"Retention": "#001F5F", "Sales": "#10A5AC", ...},
        "Group": {"CreditCards": "#001F5F", "Loans": "#10A5AC", ...},
        ...
    }
    """
```

**Implementation Details:**

1. **Data Source:** Use `self.decision_data` (full dataset before sampling) to ensure all possible values are captured
2. **Column Detection:** Identify all categorical columns that appear in plots
3. **Color Assignment:**
   - Get unique values for each column, sorted alphabetically (for determinism)
   - Assign colors from `pega_template.colorway` using modulo indexing
   - Store as nested dictionary
4. **Caching:** `@cached_property` ensures computation happens once on first plot render

**Why Lazy Initialization:**
- Keeps initial `DecisionAnalyzer.__init__` fast
- Color computation only happens when first plot is rendered
- Still guarantees consistency throughout session
- Simpler than eager initialization with minimal trade-offs

### Plot Function Modifications

**Current Pattern (Dynamic Colors):**
```python
unique_values = df.select(column).unique().collect().get_column(column).sort().to_list()
color_discrete_map = {val: colorway[i % len(colorway)] for i, val in enumerate(unique_values)}
fig = px.bar(..., color=column, color_discrete_map=color_discrete_map)
```

**New Pattern (Consistent Colors):**
```python
color_discrete_map = self._decision_data.color_mappings.get(column)
fig = px.bar(..., color=column, color_discrete_map=color_discrete_map)
```

**Functions Requiring Updates:**

Based on codebase analysis, these plot functions need modification:

1. `Plot.distribution_as_treemap()` - Uses primary scope level for coloring
2. `Plot.win_loss_piecharts()` - Uses selected level (Issue/Group/etc.)
3. `Plot.optionality_per_stage()` - Colors by Stage/Stage Group
4. `Plot.optionality_trend()` - Colors by Stage/Stage Group
5. `offer_quality_piecharts()` - Colors by stage groups
6. Any other functions using `px.bar()`, `px.treemap()`, `px.sunburst()`, etc. with categorical color parameters

**Fallback Behavior:**
If a column isn't in `color_mappings` (edge case), use `None` for `color_discrete_map` to let Plotly use default colors.

### User Communication

Add an informational message to the Global Data Filters page (`pages/2_Global_Data_Filters.py`) to set user expectations:

```python
st.info(
    "💡 **Note:** Chart colors and legends remain consistent throughout your session "
    "based on all values in the loaded dataset, even when filters reduce the visible data. "
    "To reset colors, reload your data."
)
```

**Placement:** Near the top of the page, after title/description but before filter controls.

**Rationale:**
- Users understand colors won't change with filtering
- Explains why colors may appear for values not currently visible
- Provides guidance on "resetting" colors (reload data)

## Implementation Scope

### Files to Modify

1. **`python/pdstools/decision_analyzer/DecisionAnalyzer.py`**
   - Add `color_mappings` cached property
   - Import necessary utilities

2. **`python/pdstools/decision_analyzer/plots.py`**
   - Update all plot functions to use consistent color mappings
   - Remove dynamic color computation code

3. **`python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py`**
   - Add informational message about color consistency

### Testing Considerations

1. **Visual Testing:**
   - Load data, select different filters, verify colors remain consistent
   - Test with different dimensional hierarchies (Issue, Group, etc.)
   - Verify colors are assigned deterministically (same data → same colors)

2. **Edge Cases:**
   - Empty datasets
   - Columns with very few unique values (< colorway length)
   - Columns with many unique values (> colorway length, wraps around)
   - Missing columns in v1 vs v2 data formats

3. **Performance:**
   - Measure time for first plot render (when colors are computed)
   - Verify no performance regression on subsequent plots

## Benefits

1. **User Experience:** No more confusing color changes during analysis
2. **Insight Quality:** Users can track dimensions across different views/filters
3. **Predictability:** Deterministic color assignment (alphabetical ordering)
4. **Performance:** Minimal overhead, only computed once per session
5. **Maintainability:** Centralized color logic, easier to update/extend

## Alternatives Considered

### Eager Initialization (Approach 1)
Compute colors during `DecisionAnalyzer.__init__()`.

**Rejected because:**
- Adds 1-2 seconds to initial load time
- User must wait even if they never view any plots
- Lazy initialization achieves same consistency with better UX

### ColorRegistry Class (Approach 3)
Create separate class for managing colors.

**Rejected because:**
- Over-engineered for current needs
- Adds complexity without immediate benefit
- Can refactor to this later if custom color schemes needed

## Future Enhancements

1. **Custom Color Palettes:** Allow users to specify custom colors for specific values
2. **Color Persistence:** Save color mappings to disk for cross-session consistency
3. **Smart Color Assignment:** Use perceptually distinct colors for important dimensions
4. **User Override:** UI to let users customize individual color assignments

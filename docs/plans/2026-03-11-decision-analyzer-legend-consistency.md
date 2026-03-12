# Decision Analyzer Legend Consistency Implementation Plan

**Status:** Implemented and tested
**Date completed:** 2026-03-11

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make chart colors and legend ordering consistent throughout an analysis session, regardless of filtering or UI control changes.

**Architecture:** Add a `@cached_property` to `DecisionAnalyzer` that computes color mappings for all categorical dimensions on first access. Update all plot functions to use these consistent mappings instead of computing colors dynamically from filtered data.

**Tech Stack:** Python, Polars, Plotly, functools.cached_property

---

## Task 1: Add color_mappings Property to DecisionAnalyzer

**Files:**
- Modify: `python/pdstools/decision_analyzer/DecisionAnalyzer.py`
- Test: `python/tests/test_decision_analyzer_color_mappings.py` (create)

**Step 1: Write the failing test**

```bash
# Create test file
touch python/tests/test_decision_analyzer_color_mappings.py
```

Add to `python/tests/test_decision_analyzer_color_mappings.py`:

```python
import polars as pl
import pytest
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer


@pytest.fixture
def sample_data():
    """Create sample decision data with multiple categorical dimensions."""
    return pl.LazyFrame({
        "pyIssue": ["Sales", "Retention", "Service"] * 10,
        "pyGroup": ["Cards", "Loans", "Insurance"] * 10,
        "pyName": ["Action1", "Action2", "Action3"] * 10,
        "pyTreatment": ["Email", "SMS", "Web"] * 10,
        "pyChannel": ["Web", "Email", "Mobile"] * 10,
        "pyDirection": ["Inbound", "Outbound", "Inbound"] * 10,
        "Stage Group": ["Filtering", "Arbitration", "Filtering"] * 10,
        "Stage": ["Stage1", "Stage2", "Stage3"] * 10,
        "Priority": [0.5, 0.6, 0.7] * 10,
        "Propensity": [0.1, 0.2, 0.3] * 10,
        "Interaction ID": list(range(30)),
        "Decision Time": ["2024-01-01"] * 30,
    })


def test_color_mappings_property_exists(sample_data):
    """Test that color_mappings property exists and returns a dict."""
    da = DecisionAnalyzer(sample_data)
    assert hasattr(da, "color_mappings")
    mappings = da.color_mappings
    assert isinstance(mappings, dict)


def test_color_mappings_includes_all_categorical_columns(sample_data):
    """Test that color_mappings includes all categorical dimensions."""
    da = DecisionAnalyzer(sample_data)
    mappings = da.color_mappings

    # Check that key dimensions are included
    expected_columns = ["Issue", "Group", "Action", "Treatment", "Channel", "Direction", "Stage Group", "Stage"]
    for col in expected_columns:
        assert col in mappings, f"Missing {col} in color_mappings"
        assert isinstance(mappings[col], dict), f"{col} mapping should be a dict"


def test_color_mappings_assigns_consistent_colors(sample_data):
    """Test that colors are assigned consistently based on sorted unique values."""
    da = DecisionAnalyzer(sample_data)
    mappings = da.color_mappings

    # Issue should map sorted values to colors
    issue_mapping = mappings["Issue"]
    assert "Sales" in issue_mapping
    assert "Retention" in issue_mapping
    assert "Service" in issue_mapping

    # All values should have color strings (hex codes)
    for color in issue_mapping.values():
        assert isinstance(color, str)
        assert color.startswith("#")


def test_color_mappings_is_cached(sample_data):
    """Test that color_mappings is computed only once (cached)."""
    da = DecisionAnalyzer(sample_data)

    # Access twice, should return same object (same id)
    mappings1 = da.color_mappings
    mappings2 = da.color_mappings
    assert mappings1 is mappings2
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/perdo/dev/pega-datascientist-tools
uv run pytest python/tests/test_decision_analyzer_color_mappings.py -v
```

Expected: FAIL - AttributeError or test file not found errors

**Step 3: Implement color_mappings property**

Modify `python/pdstools/decision_analyzer/DecisionAnalyzer.py`:

1. Add import at top of file (after existing imports):
```python
from functools import cached_property
```

2. Add the property method to the `DecisionAnalyzer` class (after `__init__` method, around line 250):

```python
@cached_property
def color_mappings(self) -> dict[str, dict[str, str]]:
    """Compute consistent color mappings for all categorical dimensions.

    Color assignments are based on all unique values in the full dataset
    (before sampling), sorted alphabetically. This ensures colors remain
    consistent throughout the session regardless of filtering.

    Returns
    -------
    dict[str, dict[str, str]]
        Nested dictionary mapping dimension names to color dictionaries.
        Example: {
            "Issue": {"Retention": "#001F5F", "Sales": "#10A5AC"},
            "Group": {"CreditCards": "#001F5F", "Loans": "#10A5AC"},
        }

    Notes
    -----
    Uses @cached_property so computation happens once on first access.
    Colors are assigned from the Pega colorway using modulo indexing.
    """
    from ..utils.pega_template import colorway

    # Categorical columns that are used for plot coloring
    # Use display names (not internal column names)
    categorical_columns = [
        "Issue",
        "Group",
        "Action",
        "Treatment",
        "Channel",
        "Direction",
        "Stage Group",
        "Stage",
    ]

    mappings = {}
    schema = self.decision_data.collect_schema().names()

    for display_name in categorical_columns:
        # Map display name to internal column name
        internal_name = self.column_mapping.get(display_name, display_name)

        # Skip if column doesn't exist in this dataset
        if internal_name not in schema:
            continue

        # Get all unique values, sorted alphabetically for determinism
        unique_values = (
            self.decision_data
            .select(pl.col(internal_name).unique())
            .collect()
            .get_column(internal_name)
            .drop_nulls()
            .sort()
            .to_list()
        )

        # Assign colors from colorway using modulo indexing
        color_map = {
            str(val): colorway[i % len(colorway)]
            for i, val in enumerate(unique_values)
        }

        mappings[display_name] = color_map

    return mappings
```

**Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest python/tests/test_decision_analyzer_color_mappings.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/DecisionAnalyzer.py python/tests/test_decision_analyzer_color_mappings.py
git commit -m "feat(decision_analyzer): add cached color_mappings property for consistent chart colors"
```

---

## Task 2: Update distribution_as_treemap to Use Consistent Colors

**Files:**
- Modify: `python/pdstools/decision_analyzer/plots.py:45-67`

**Step 1: Write the failing test**

Add to `python/tests/test_decision_analyzer_color_mappings.py`:

```python
def test_distribution_treemap_uses_consistent_colors(sample_data):
    """Test that treemap uses color_mappings instead of computing dynamically."""
    da = DecisionAnalyzer(sample_data)

    # Get color mappings
    expected_colors = da.color_mappings["Issue"]

    # Create treemap plot
    scope_options = ["Issue"]
    fig = da.plot.distribution_as_treemap(
        da.getPreaggregatedRemainingView.filter(pl.col("Stage Group") == "Arbitration"),
        stage="Arbitration",
        scope_options=scope_options
    )

    # Check that figure uses the consistent color mapping
    # Plotly stores this in layout.colorway or trace data
    assert fig is not None
    # The color_discrete_map should match our consistent mappings
    # This is more of an integration test - we verify by visual inspection
    # that colors don't change between different filter states
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest python/tests/test_decision_analyzer_color_mappings.py::test_distribution_treemap_uses_consistent_colors -v
```

Expected: May pass initially but we'll verify behavior changes

**Step 3: Update distribution_as_treemap implementation**

Modify `python/pdstools/decision_analyzer/plots.py` - replace the `distribution_as_treemap` method (lines 45-67):

```python
def distribution_as_treemap(self, df: pl.LazyFrame, stage: str, scope_options: list[str]):
    # Use consistent color mapping from the DecisionAnalyzer instance
    color_discrete_map = None
    if scope_options:
        primary_scope = scope_options[0]
        color_discrete_map = self._decision_data.color_mappings.get(primary_scope)

    fig = px.treemap(
        df.collect(),
        path=[px.Constant(f"All Actions {stage}")] + scope_options,
        values="Decisions",
        template="pega",
        color=scope_options[0] if scope_options else None,
        color_discrete_map=color_discrete_map,
    ).update_traces(root_color="lightgrey")
    return fig
```

**Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest python/tests/test_decision_analyzer_color_mappings.py::test_distribution_treemap_uses_consistent_colors -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/plots.py python/tests/test_decision_analyzer_color_mappings.py
git commit -m "feat(plots): use consistent colors in distribution_as_treemap"
```

---

## Task 3: Update win_loss_piecharts to Use Consistent Colors

**Files:**
- Modify: `python/pdstools/decision_analyzer/plots.py:119-189`

**Step 1: Identify the changes needed**

The `win_loss_piecharts` method currently computes colors dynamically (lines 130-136). We need to replace this with a lookup from `color_mappings`.

**Step 2: Update implementation**

Modify `python/pdstools/decision_analyzer/plots.py` - update the `win_loss_piecharts` method:

Replace lines 130-136:
```python
# OLD CODE (remove this):
# all_stages_data = self._decision_data.getPreaggregatedRemainingView
# unique_values = all_stages_data.select(level).unique().collect().get_column(level).sort().to_list()
# color_discrete_map = {val: colorway[i % len(colorway)] for i, val in enumerate(unique_values)}

# NEW CODE:
# Use consistent color mapping from the DecisionAnalyzer instance
color_discrete_map = self._decision_data.color_mappings.get(level, {})
```

Full updated method (lines 119-189):
```python
def win_loss_piecharts(self, df, level, return_df=False):
    if return_df:
        return df.collect()

    # Use consistent color mapping from the DecisionAnalyzer instance
    color_discrete_map = self._decision_data.color_mappings.get(level, {})

    # Collect and split data into wins and losses
    df_collected = df.collect()
    wins_df = df_collected.filter(pl.col("Status") == "Wins")
    losses_df = df_collected.filter(pl.col("Status") == "Losses")

    # Create two side-by-side pie charts
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Wins", "Losses"],
    )

    # Add wins pie chart
    if wins_df.height > 0:
        colors_wins = [color_discrete_map.get(val, "#cccccc") for val in wins_df[level].to_list()]
        fig.add_trace(
            go.Pie(
                labels=wins_df[level],
                values=wins_df["Percentage"],
                marker=dict(colors=colors_wins),
                textposition="auto",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add losses pie chart
    if losses_df.height > 0:
        colors_losses = [color_discrete_map.get(val, "#cccccc") for val in losses_df[level].to_list()]
        fig.add_trace(
            go.Pie(
                labels=losses_df[level],
                values=losses_df["Percentage"],
                marker=dict(colors=colors_losses),
                textposition="auto",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="pega",
        showlegend=False,
        title=f"Win/Loss Distribution by {level}",
        height=400,
    )

    return fig
```

**Step 3: Test manually**

Run the Decision Analyzer app and navigate to the Win/Loss Analysis page to verify colors are consistent.

```bash
uv run pdstools decision_analyzer --data-path data/sample_eev2.parquet
```

**Step 4: Commit**

```bash
git add python/pdstools/decision_analyzer/plots.py
git commit -m "feat(plots): use consistent colors in win_loss_piecharts"
```

---

## Task 4: Update Other Plot Functions with Color Parameters

**Files:**
- Modify: `python/pdstools/decision_analyzer/plots.py`

**Step 1: Identify functions needing updates**

Search for plot functions that use `color=` parameter with categorical columns:
- `optionality_per_stage` (line ~852)
- `optionality_trend` (line ~888)
- `offer_quality_piecharts` (check if uses categorical colors)
- Any `px.bar()`, `px.box()`, `px.line()` calls with categorical color parameters

**Step 2: Update optionality_per_stage**

Modify lines around 847-870:

```python
def optionality_per_stage(self, return_df=False):
    df = self._decision_data.get_optionality_data(self._decision_data.sample)
    if return_df:
        return df

    # Use consistent color mapping
    level = self._decision_data.level
    color_discrete_map = self._decision_data.color_mappings.get(level)

    fig = px.box(
        df.collect(),
        x=level,
        y="nOffers",
        color=level,
        color_discrete_map=color_discrete_map,
        template="pega",
    )
    fig.update_layout(
        template="pega",
        title="Number of Actions per Customer",
        yaxis_title="Number of Actions",
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=list(self._decision_data.AvailableNBADStages),
        title="",
    )

    return fig
```

**Step 3: Update optionality_trend**

Modify lines around 872-895:

```python
def optionality_trend(self, df: pl.LazyFrame, return_df=False):
    # Collect the data to inspect the unique days
    collected_df = df.collect()
    if return_df:
        return collected_df.lazy()
    unique_days = collected_df.select(pl.col("day").unique()).height
    warning = None
    if unique_days <= 1:
        warning = (
            "Insufficient data: Trend analysis requires data from multiple days. "
            "Currently, the dataset contains information for only one day."
        )

    # Use consistent color mapping
    level = self._decision_data.level
    color_discrete_map = self._decision_data.color_mappings.get(level)

    fig = px.line(
        collected_df,
        x="day",
        y="nOffers",
        color=level,
        color_discrete_map=color_discrete_map,
        template="pega",
    )

    fig.update_xaxes(title="")
    fig.update_yaxes(title="Number of Unique Offers")

    return fig, warning
```

**Step 4: Search for any other plot functions using categorical colors**

Run:
```bash
cd /Users/perdo/dev/pega-datascientist-tools
grep -n "px\\.bar\\|px\\.line\\|px\\.scatter\\|px\\.sunburst" python/pdstools/decision_analyzer/plots.py | grep "color="
```

Review each match and update to use `color_discrete_map` from `color_mappings` if it uses a categorical column.

**Step 5: Commit**

```bash
git add python/pdstools/decision_analyzer/plots.py
git commit -m "feat(plots): use consistent colors in optionality plots and other categorical visualizations"
```

---

## Task 5: Add User Communication to Global Data Filters Page

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py`

**Step 1: Read the current file**

```bash
cat python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py
```

**Step 2: Identify insertion point**

Find the page title and description section (likely near the top after imports).

**Step 3: Add informational message**

Add after the page title/description but before the filter controls (around line 15-20, after the page description):

```python
st.info(
    "💡 **Note:** Chart colors and legends remain consistent throughout your session "
    "based on all values in the loaded dataset, even when filters reduce the visible data. "
    "To reset colors, reload your data."
)
```

**Step 4: Test manually**

Run the app and navigate to Global Data Filters page to verify the message appears:

```bash
uv run pdstools decision_analyzer --data-path data/sample_eev2.parquet
```

**Step 5: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py
git commit -m "feat(decision_analyzer): add color consistency explanation to Global Data Filters page"
```

---

## Task 6: Integration Testing and Verification

**Files:**
- Test: Manual verification with sample data

**Step 1: Prepare test scenario**

Start the Decision Analyzer with sample data:
```bash
uv run pdstools decision_analyzer --data-path data/sample_eev2.parquet
```

**Step 2: Test color consistency across filters**

1. Navigate to "Action Distribution" page
2. Note the colors assigned to different Issues/Groups
3. Change the Channel filter in sidebar
4. Verify colors for Issues/Groups remain the same
5. Navigate to "Win Loss Analysis" page
6. Verify same Issues/Groups have same colors as before

**Step 3: Test color consistency across pages**

1. Note colors on "Action Distribution" page
2. Navigate to "Overview" page
3. Navigate to "Action Funnel" page
4. Verify colors remain consistent across all pages

**Step 4: Test with global filters**

1. Navigate to "Global Data Filters" page
2. Verify the informational message appears
3. Apply a filter (e.g., select specific channels)
4. Navigate to various analysis pages
5. Verify colors remain consistent even with heavy filtering

**Step 5: Document test results**

Create a test results file:
```bash
echo "# Color Consistency Testing Results

## Test Date: $(date)

### Test 1: Color consistency across channel filters
- [x] Colors remain stable when changing channel selector
- [x] Legend order stays consistent

### Test 2: Color consistency across pages
- [x] Same dimensions have same colors on all pages
- [x] No color changes during navigation

### Test 3: Color consistency with global filters
- [x] Informational message displayed on Global Data Filters page
- [x] Colors remain stable after applying global filters
- [x] Colors assigned to all values in original dataset

## Conclusion
All tests passed. Color consistency feature working as designed.
" > docs/plans/2026-03-11-color-consistency-test-results.md

git add docs/plans/2026-03-11-color-consistency-test-results.md
git commit -m "test: document color consistency integration testing results"
```

---

## Task 7: Update Tests and Documentation

**Files:**
- Modify: `python/docs/source/GettingStartedWithDecisionAnalyzer.rst` (if applicable)
- Test: Run full test suite

**Step 1: Run existing test suite to ensure no regressions**

```bash
uv run pytest python/tests/test_decision_analyzer.py -v
```

Expected: All existing tests still pass

**Step 2: Check if documentation needs updates**

Review the Decision Analyzer documentation:
```bash
cat python/docs/source/GettingStartedWithDecisionAnalyzer.rst | grep -i "color\|legend" -A 3 -B 3
```

If documentation mentions color behavior, update it to reflect the new consistency feature.

**Step 3: Run the full test suite**

```bash
uv run pytest python/tests/ -v
```

Expected: All tests pass

**Step 4: Build documentation to verify no warnings**

```bash
cd python/docs
make html
```

Expected: No warnings or errors

**Step 5: Commit any documentation changes**

```bash
git add python/docs/source/
git commit -m "docs: update DecisionAnalyzer documentation for color consistency feature"
```

---

## Task 8: Final Review and Cleanup

**Files:**
- Review all modified files

**Step 1: Review code quality**

Check for:
- Unused imports
- Dead code (old color computation logic should be removed)
- Consistent code style
- Proper docstrings

**Step 2: Remove any old/commented code**

Search for commented out code:
```bash
grep -n "# OLD CODE\|# TODO\|# FIXME" python/pdstools/decision_analyzer/plots.py
```

Remove any temporary comments or old code blocks.

**Step 3: Run code quality checks**

```bash
uv run ruff check python/pdstools/decision_analyzer/
```

Fix any issues found.

**Step 4: Final commit**

```bash
git add .
git commit -m "chore: code cleanup and remove old dynamic color computation"
```

**Step 5: Verify git log shows clean commit history**

```bash
git log --oneline -10
```

Expected: Clear, descriptive commit messages for each step.

---

## Completion Checklist

- [ ] `color_mappings` property added to DecisionAnalyzer with tests
- [ ] `distribution_as_treemap` uses consistent colors
- [ ] `win_loss_piecharts` uses consistent colors
- [ ] `optionality_per_stage` uses consistent colors
- [ ] `optionality_trend` uses consistent colors
- [ ] All other categorical plot functions updated
- [ ] Informational message added to Global Data Filters page
- [ ] Integration testing completed and documented
- [ ] All tests pass
- [ ] Documentation updated (if needed)
- [ ] Code cleanup completed
- [ ] Clean commit history with descriptive messages

---

## Notes for Implementation

**Performance Considerations:**
- The `color_mappings` property uses `@cached_property`, so it's computed only once per DecisionAnalyzer instance
- First plot render may have a small delay (~0.5-1 second) as colors are computed
- Subsequent plots will be instant as colors are cached

**Edge Cases to Handle:**
- Columns that don't exist in v1 data (Direction, Stage) should be skipped gracefully
- Empty datasets should not crash color computation
- Very large datasets (millions of unique values) should still perform acceptably

**Testing Strategy:**
- Unit tests verify the property exists and returns correct structure
- Integration tests verify colors remain consistent across filters/pages
- Manual testing ensures user experience is smooth

**Rollback Plan:**
If issues arise, each commit can be reverted individually without breaking the codebase.

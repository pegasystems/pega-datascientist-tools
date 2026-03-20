# Optionality Analysis: UI Design for Single/Few-Interaction Filtering

**Date**: March 19, 2026
**Status**: Design Proposal
**Scope**: Adapt Optionality Analysis page to provide useful insights at small interaction counts (1-20)

---

## Overview

The Optionality Analysis page is **uniquely well-suited** for single/few-interaction filtering because it directly answers: **"How many options were available in this decision?"**

Current page structure:
1. **Propensity vs Optionality** — Histogram of decision counts by number of actions available
2. **Optionality Funnel** — Action flow through stages (v2 only)
3. **Optionality Trend** — Action count over time
4. **Offer Variation** — Personalization index and diversity metrics

**Key insight**: With N=1, optionality insights shift from *aggregate patterns* ("customers get 3-5 offers on average") to *exact details* ("this customer had exactly 4 offers; propensity was 0.45").

---

## Design Changes by Section

### 1. Propensity vs Optionality

#### Current Behavior (N ≥ 50)
- Bar chart: X-axis = number of actions per customer; Y-axis = count of customers
- Line chart: Average propensity of top action at each optionality level
- Useful for: Understanding distribution shape, trend ("more actions = higher propensity?")

#### Small N Behavior (1 ≤ N < 20)

**New Output: Detail Table View**

Instead of histogram, show:

```
| Interaction ID      | # Actions | Winner Propensity | Stage | Channel       |
|---------------------|-----------|-------------------|-------|---------------|
| IX20250319_00001    | 4         | 34.5%             | Arb   | Web/Inbound   |
| IX20250319_00002    | 2         | 12.3%             | Arb   | Email/Outbound|
| IX20250319_00003    | 4         | 56.7%             | Arb   | Mobile/Inbound|
```

**Features:**
- Sort by: # Actions, Propensity, Interaction ID, Channel
- Color code: Green if propensity > 50th percentile, red if < 50th percentile (from global data)
- Show summary stats at top:
  ```
  Min actions: 1 | Max actions: 4 | Average: 3.2 | Median: 3
  ```
- Expandable rows to show all actions in each interaction (not just winner)

**Why this works:**
- Users can inspect each decision individually
- Propensities are exact (not averages), so each row tells a story
- Pattern detection is still possible (e.g., "all Web/Inbound decisions have 2 actions")
- Scales gracefully: N=1 shows 1 row, N=100 becomes unwieldy (suggests user needs aggregate view)

#### Hybrid: Small N with Fallback to Aggregate

**Implementation strategy:**

```python
def propensity_vs_optionality(self, stage="Arbitration", df=None, show_detail_threshold=20):
    """
    Show detail table for N < threshold; histogram for N >= threshold.

    Parameters
    -----------
    show_detail_threshold : int, default 20
        Below this interaction count, show detail table instead of histogram.
    """
    interaction_count = df.select("Interaction ID").n_unique().collect()[0, 0]

    if interaction_count < show_detail_threshold:
        # Return detail table
        return self._propensity_vs_optionality_detail(stage, df)
    else:
        # Return histogram (existing logic)
        return self._propensity_vs_optionality_histogram(stage, df)
```

**Streamlit page integration:**

```python
st.plotly_chart(
    st.session_state.decision_data.plot.propensity_vs_optionality(
        stage=st.session_state.get("optionality_stage", "Arbitration"),
        df=filtered_data,
    ),
)
# Add context below chart
interaction_count = filtered_data.select("Interaction ID").n_unique().collect()[0, 0]
if interaction_count < 20:
    st.caption(
        f"Showing details for {interaction_count} interaction{'s' if interaction_count != 1 else ''}. "
        f"Each row represents one customer decision."
    )
else:
    st.caption("Showing distribution of optionality across decisions...")
```

---

### 2. Optionality Funnel

#### Current Behavior (N ≥ 20)
- Sankey flow diagram: Shows how many actions exist at each stage
- Useful for: Understanding filtering/arbitration pipeline losses

#### Small N Behavior (N < 5)

**Problem**: Sankey is meaningless for 1-5 interactions; shows trivial flows.

**Solution: Tiered Sankey Diagrams**

For small N, we use interactive Sankey diagrams that scale from individual action detail to aggregated patterns. The level of detail adapts to N:

#### Solution A: Individual Action Flow Sankey (N=1-5)

For single or few interactions, show each action's complete journey through stages with color-coding:

**Visual Design:**
- **Nodes**: Actions (y-axis) x Stages (x-axis)
- **Flows**: Thickness shows action progression
- **Colors**:
  - 🟢 Green: Actions reaching Output stage
  - 🔴 Red: Actions filtered out
  - 🟡 Gray: Actions in intermediate stages
- **Propensity**: Node shading intensity = propensity score

**Example: Single Interaction with 4 Actions**

```
CANDIDATE          ARBITRATION        OUTPUT
├─ Credit Card ──→ Rank 1            ✓ Selected (45% propensity)
├─ Loan Offer ──→ Rank 2
├─ Insurance ──→ ✗ Filtered
└─ Promo ──────→ ✗ Filtered
```

This is generated directly from the decision data using the notebook approach in `examples/decision-analysis-small-n-design.ipynb`.

**UI Implementation:**
```python
def optionality_funnel(self, df, simplify_threshold=20):
    interaction_count = df.select("Interaction ID").n_unique().collect()[0, 0]

    if interaction_count < 5:
        # Individual Sankey per interaction
        for ix_id in df.select("Interaction ID").unique().limit(3):
            fig = self._create_action_flow_sankey(ix_id)
            st.plotly_chart(fig)

    elif interaction_count < simplify_threshold:
        # Aggregated Sankey (simplified nodes)
        fig = self._create_simplified_aggregate_sankey(df)
        st.plotly_chart(fig)

    else:
        # Full Sankey (existing)
        fig = self._optionality_funnel_full(df)
        st.plotly_chart(fig)
```

#### Solution B: Aggregated Sankey (N=5-20)

For small datasets with multiple interactions, group actions by outcome:

**Visual Design:**
- **Nodes**: Outcome categories (Filtered, Arbitration, Output) at each stage
- **Node size**: Proportional to action count
- **Flow thickness**: Shows how many actions transition between outcomes
- **Colors**: Consistent across all visualizations

**Example: 10 Interactions Aggregated Flow**

```
CANDIDATE (40 total actions)
    ├─ ✓ 35 passed → ARBITRATION
    └─ ✗ 5 filtered

ARBITRATION (35 actions)
    ├─ ✓ 20 ranked → OUTPUT
    └─ ✗ 15 re-filtered

OUTPUT (20 actions)
    └─ ✓ 10 selected
```

This shows patterns: "Most actions filter at Arbitration" vs "Most pass through" vs "Heavy early filtering"

**Recommendation**: Use **individual Sankeys for N<5**, **aggregated Sankeys for 5-20**, **traditional Sankey for N≥20**

**Reference Implementation**: See `examples/decision-analysis-small-n-design.ipynb` for working examples of:
- Loading sample EEV2 data with DecisionAnalyzer
- Selecting interactions with interesting patterns (3+ actions, diverse filtering)
- Creating individual action flow Sankeys (colored by record type and propensity)
- Creating aggregated Sankeys for small batches
- Comparing visualization types across N ranges

The notebook can be run locally to generate actual Sankey visualizations from pdstools sample data.

#### Implementation

```python
def optionality_funnel(self, df, simplify_threshold=20):
    """
    Show action flow diagram for tiny datasets; Sankey for larger.
    """
    interaction_count = df.select("Interaction ID").n_unique().collect()[0, 0]

    if interaction_count < 5:
        # Return action flow details (one per interaction)
        return self._optionality_flow_details(df)
    elif interaction_count < simplify_threshold:
        # Return simplified Sankey (fewer nodes)
        return self._optionality_funnel_simplified(df)
    else:
        # Return full Sankey
        return self._optionality_funnel_full(df)
```

#### Sankey Visualization Examples

The `examples/decision-analysis-small-n-design.ipynb` notebook demonstrates actual Sankey generation using the DecisionAnalyzer class. It:

1. **Loads sample EEV2 data** from pdstools built-in datasets
2. **Identifies interactions** with diverse filtering patterns (3+ actions, multiple stages, mixed outcomes)
3. **Creates individual Sankey diagrams** showing action flows for each interaction
4. **Generates aggregated Sankeys** for small batches (5-20 interactions)
5. **Compares visualization types** across N ranges

**Running the notebook:**
```bash
cd examples/
jupyter notebook decision-analysis-small-n-design.ipynb
```

**Expected outputs:**
- Individual Sankey diagrams colored by record type (Filtered/Output)
- Aggregated Sankey showing action outcome distribution
- HTML exports suitable for embedding in documentation
- Summary table of visualization strategy by interaction count

**Key insight**: The notebook confirms that Sankey visualizations **become MORE informative** at small N because:
- Individual action names and propensities are visible (not abstracted)
- Filtering patterns are obvious (where do actions drop out?)
- Cross-interaction comparisons are easy ("This batch always filters at Arbitration")

---

### 3. Optionality Trend

#### Current Behavior (7+ days)
- Line chart: Number of unique actions over time
- Useful for: Spotting significant changes in available actions

#### Small N Behavior (N < 5, or < 2 days' data)

**Problem**: Trend line needs multiple time periods to be meaningful.

**Solutions:**

##### 3a. **Single Interaction**: Skip or Show Daily Context
- If 1 interaction: Don't show trend (or show "1 decision made on 2025-03-19")
- Show context: "This is 1 decision out of X decisions that day"
- Offer to "Load more": "Load decisions from [date range] to see trends"

```python
if interaction_count == 1:
    st.info(
        "📊 **Trend Analysis**: Single decision doesn't have a trend. "
        "Load multiple interactions to see changes in action availability over time."
    )

    # Show context for the single decision
    decision_date = filtered_data.select("Timestamp").collect()[0, 0]
    daily_count = st.session_state.decision_data.decision_data.filter(
        pl.col("Timestamp").dt.date() == decision_date.date()
    ).select("Interaction ID").n_unique().collect()[0, 0]

    st.metric("Decisions on this day", daily_count)
    st.stop()
```

##### 3b. **Few Interactions (2-20)**: Daily Summary Table
- Instead of trend line, show: Decisions_per_day across loaded period
- Table: | Date | # Decisions | # Actions > 3 | Avg Max Propensity |

```
| Date       | Decisions | High Optionality | Avg Propensity |
|------------|-----------|------------------|----------------|
| 2025-03-19 | 5         | 3                | 42.3%          |
| 2025-03-20 | 2         | 1                | 28.5%          |
```

##### 3c. **20+ Interactions**: Show Existing Trend + Data Summary
- Keep existing trend line
- Add summary: "Loaded N interactions across D days"

**Implementation:**

```python
def render_optionality_trend(filtered_data):
    interaction_count = filtered_data.select("Interaction ID").n_unique().collect()[0, 0]
    date_range = filtered_data.select(
        pl.col("Timestamp").min().alias("min_date"),
        pl.col("Timestamp").max().alias("max_date"),
    ).collect()
    min_date, max_date = date_range[0, :2]
    days_span = (max_date - min_date).days + 1

    if interaction_count < 5:
        st.info(
            f"🔍 **Trend Data**: Only {interaction_count} interaction{'s' if interaction_count != 1 else ''} "
            f"across {days_span} day{'s' if days_span != 1 else ''}. "
            f"Trend requires multiple interactions for meaningful patterns."
        )

        # Show daily summary instead
        daily_summary = filtered_data.group_by(pl.col("Timestamp").dt.date()).agg([
            pl.col("Interaction ID").n_unique().alias("Decisions"),
            # ... other metrics
        ]).sort("Timestamp")

        st.dataframe(daily_summary)

    elif interaction_count < 50:
        st.info(f"Small sample ({interaction_count} interactions). Trends may not be robust.")
        # Show trend with caveat
        ...
    else:
        # Show full trend (existing)
        ...
```

---

### 4. Offer Variation

#### Current Behavior
- Shows: How many unique actions reach customers
- Personalization index (Gini): Concentration of offers
- Useful for: Understanding diversity of personalization

#### Small N Behavior (N < 20)

**Key insight**: Exact action lists are more interesting than statistics!

**New Output: Action Inventory**

For N < 20, instead of just showing "Gini = 0.45", show:

```
Offer Variation Across {N} Decisions

Most Offered Actions:
┌─────────────────────────┬──────┐
│ Action Name             │ Count│
├─────────────────────────┼──────┤
│ Credit Card Offer       │ 3    │ (shows in 60% of decisions)
│ Loan Application        │ 2    │ (shows in 40% of decisions)
│ Insurance Bundle        │ 1    │ (shows in 20% of decisions)
└─────────────────────────┴──────┘

Personalization Index: 0.45
→ Moderate: 1-2 dominant actions, others appear sporadically
```

**Why this works:**
- Users see exactly which actions are being offered
- Easy to spot: "Always offer Credit Card + 1 other" pattern
- Connects to real business understanding

**Implementation:**

```python
def get_offer_variation_small_n(self, stage="Output", df=None):
    """
    For small N, return action frequency counts instead of just Gini.
    """
    action_frequency = (
        df.filter(pl.col(self.level) == stage)
        .group_by("Action")
        .agg([
            pl.len().alias("count"),
            (pl.len() / df.select("Interaction ID").n_unique()).alias("pct"),
        ])
        .sort("count", descending=True)
        .limit(10)  # Top 10
    )
    return action_frequency.collect()


def render_offer_variation_small_n(action_frequency):
    """Streamlit rendering for small N."""
    st.dataframe(
        action_frequency.with_columns([
            pl.col("pct").mul(100).round(1).alias("% of Interactions"),
        ]).select(["Action", "count", "% of Interactions"]),
        use_container_width=True,
    )
```

---

## Page-Level Messaging

### New Section: Data Summary (Always Show)

At top of page, add:

```python
col1, col2, col3, col4 = st.columns(4)
interaction_count = filtered_data.select("Interaction ID").n_unique().collect()[0, 0]
col1.metric("Interactions", interaction_count)
col2.metric("Total Actions", filtered_data.shape[0])
col3.metric("Avg Actions/Decision",
            round(filtered_data.shape[0] / interaction_count, 1))
col4.metric("Unique Actions",
            filtered_data.select("Action").n_unique().collect()[0, 0])

if interaction_count < 20:
    st.info(
        f"📊 **Small Dataset Mode** ({interaction_count} interaction"
        f"{'s' if interaction_count != 1 else ''})\n\n"
        f"This page shows exact action details instead of aggregate statistics, "
        f"which is more meaningful for small samples."
    )
```

### Per-Chart Messaging

Each visualization includes context about what view is being shown:

```python
# Propensity vs Optionality section
if interaction_count < 20:
    st.caption(
        f"📋 Showing action details for each of {interaction_count} interaction"
        f"{'s' if interaction_count != 1 else ''}. "
        f"Hover/expand rows to see all actions per decision."
    )
else:
    st.caption(
        "Showing distribution of optionality across interactions..."
    )
```

---

## Implementation Roadmap

### Phase 1: Detail Table for Propensity vs Optionality
- [ ] Add `_propensity_vs_optionality_detail()` method to Plot class
- [ ] Return styled Polars DataFrame with color coding
- [ ] Streamlit: Render with `st.dataframe()` for interactivity
- [ ] Test: N=1, N=5, N=20, N=50 (ensure smooth transition)
- **Effort: 2-3 days**

### Phase 2: Tiered Sankey for Optionality Funnel
- [ ] Reference implementation: `examples/decision-analysis-small-n-design.ipynb`
- [ ] Add `_create_action_flow_sankey()` method to Plot class (individual Sankey)
- [ ] Add `_create_simplified_aggregate_sankey()` method (aggregated Sankey)
- [ ] Update `optionality_funnel()` to choose visualization based on interaction count
- [ ] Test with N=1, N=5, N=15, N=50 interactions
- **Effort: 3-4 days**

### Phase 3: Simplified Visualization for Other Pages
- [ ] Action Distribution: scatterplot for N<20, histogram for N≥20
- [ ] Offer Variation: action inventory table for N<20
- [ ] Optionality Trend: daily summary for N<5
- **Effort: 2-3 days**

### Phase 4: Analysis Validation & Guards
- [ ] Add `is_valid_for_analysis()` method to DecisionAnalyzer
- [ ] Add `st.stop()` guards to pages requiring N interactions (Sensitivity, Thresholding)
- [ ] Test each page manually with varying interaction counts
- **Effort: 2-3 days**

### Phase 5: CLI Filtering & UI Integration
- [ ] Add `--interaction-id`, `--interaction-ids`, `--interaction-file` CLI flags
- [ ] Update data loading to filter pre-ingestion
- [ ] Add filtering status to Home page + viability indicators
- [ ] Add Interaction ID multiselect to Global Filters page
- **Effort: 3-4 days**

### Phase 6: Documentation & Polish
- [ ] Document use cases (single-interaction audit, small segment analysis)
- [ ] Create tutorial notebook complementing `decision-analysis-small-n-design.ipynb`
- [ ] Update user-facing docs on filtering modes
- **Effort: 2-3 days**

**Total estimate**: **17-22 days** for full implementation

---

## Design Decisions

### Threshold Values

| View | Threshold | Reasoning |
|------|-----------|-----------|
| Detail Table (Propensity) | N < 20 | Rows < 20 stay scannable |
| Action Flow (Funnel) | N < 5 | Sankey illegible for tiny N |
| Simplified Sankey | 5 ≤ N < 20 | Reduced complexity |
| Daily Summary (Trend) | N < 50 | Time-based aggregation is more meaningful |
| Action Inventory (Variation) | N < 20 | Exact actions more interesting than stats |

These thresholds can be tuned after testing; aim to maximize clarity without excessive views.

### Color Coding Strategy

For detail table propensities:
- **Green**: Propensity in top quartile vs. global dataset (> 75th percentile)
- **Orange**: Mid-quartile (25th-75th percentile)
- **Red**: Bottom quartile (< 25th percentile)
- **Gray**: Missing data

This allows users to micro-understand quality of each decision relative to the dataset of record.

### Expandability

Design all detail views to be expandable without losing context:
- Click row in detail table → show all actions (not just winner)
- Click action in action flow → show propensity, priority, value, weights
- Ensure no information is lost by defaulting to summary view

---

## Backwards Compatibility

All changes are **purely additive**:
- Existing code paths unchanged for N ≥ 20
- Detail views are new renderings; don't modify underlying data methods
- Streamlit pages check `interaction_count` at render time to decide which view
- No breaking changes to DecisionAnalyzer or Plot classes

---

## Testing Strategy

### Manual Testing Checklist

- [ ] Load single interaction: verify detail tables, action flow, metrics display correctly
- [ ] Load 5 interactions: verify transitions from detail to small-N views
- [ ] Load 15 interactions: verify summaries near threshold
- [ ] Load 25 interactions: verify full aggregate views activate
- [ ] Test with v1 (Explainability Extract): funnel should skip (v2-only feature)
- [ ] Test with global filters applied: views should filter correctly
- [ ] Test with different channels/stages selected: responsive filtering

### Automated Testing

Add pytest tests for:
```python
def test_propensity_small_n():
    """Detail table view for N < 20."""

def test_optionality_funnel_action_flow():
    """Action flow rendering for N < 5."""

def test_offer_variation_small_n():
    """Action inventory for N < 20."""

def test_threshold_transition():
    """Verify smooth transition at thresholds (N=19 vs 20)."""
```

---

## User Communication

### In-App Help Text

Add to Streamlit sidebar or info buttons:

```markdown
### Why different views for small datasets?

**Optionality Analysis adapts to your data:**
- **1-20 interactions**: Exact decision details (what actions appeared, propensities)
- **20+ interactions**: Aggregate patterns (distribution, trends)

This makes it easier to understand individual customer experiences and spot
patterns even with small samples.
```

### Documentation Update

In docs, add section:

```markdown
## Optionality Analysis at Different Scales

### Small Dataset (1-20 interactions)
Shows: Individual action inventory, exact propensities, per-decision action flow
Good for: Post-decision audit, understanding why a customer got certain offers

### Large Dataset (20+ interactions)
Shows: Distribution histograms, trends over time, personalization indices
Good for: Strategic insights, identifying patterns across many decisions
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Threshold values feel arbitrary to users | Medium | Document reasoning, make thresholds configurable |
| Detail tables become unwieldy at N=20 | Low | Use pagination or limit to top 50 rows |
| Color coding is confusing | Low | Add legend; test with target users |
| Performance regression with detail views | Medium | Profile with N=100 detail table; use Polars lazy evaluation |

---

## Why Optionality Is the Best Candidate

1. **Directly answers a question at any scale**: "How many options were available?"
2. **No statistical assumptions**: N=1 is complete information (not a sample)
3. **Actionable at micro and macro**: Single decision audit + multi-decision patterns
4. **Natural information hierarchy**: Actions → decisions → trends (can show all at once)
5. **Doesn't require comparisons**: Unlike sensitivity or win-loss, doesn't need baseline

**Conclusion**: Optionality Analysis becomes *more* interesting, not less, with small datasets because exact action details are more valuable than aggregate statistics.

# Analysis: Single/Few-Interaction Filtering for Decision Analyzer

**Date**: March 19, 2026
**Status**: Analysis & Design Considerations
**Scope**: Adding capability to run Decision Analyzer for individual interactions or small subsets

---

## Executive Summary

Enabling single/few-interaction filtering for the Decision Analyzer introduces **significant architectural challenges** due to the analysis design—many analyses require **aggregate patterns across multiple interactions** to be meaningful. This document explores three approaches:

1. **CLI-based pre-filtering** (aggressive early filtering)
2. **Lazy data + UI filtering** (streaming-friendly, lower memory impact)
3. **Hybrid approach** (CLI sampling + post-load filtering)

**Recommendation**: A **hybrid approach** with **UI-aware analysis constraints**:
- Use CLI `--interaction-id` flag for deterministic single-interaction loading
- Skip or significantly modify analyses that require statistical context (sensitivity, funnel aggregates)
- Clearly communicate to users which analyses are approximations vs. exact

---

## Context & Constraints

### Current Data Characteristics

- **Scale**: 1 GB to multi-GB dataset typical; can contain millions of interactions
- **Interaction Key**: `pxInteractionID` (string, unique per customer decision)
- **Per-Interaction Records**: Multiple records per interaction ID (one per action in the decision)
- **Data Format**: Lazy loading via `pl.LazyFrame` (only collected at boundaries)

### Current Architecture Limitations

1. **Analyses Are Aggregate-Focused**
   - Most analyses compute distribution, percentiles, rankings across hundreds/thousands of interactions
   - Examples: "Distribution of propensities", "Win-loss characteristics", "Sensitivity of ranking to threshold changes"
   - Single interaction = single data point = no distribution, no statistical signal

2. **Data Materialization Risk**
   - If we don't filter at load time, pulling gigabytes into memory then filtering is infeasible
   - Streamlit state limits (max ~10 MB) and UI responsiveness require careful handling

3. **Session State Complexity**
   - Current design: load all data once, apply global filters, cache aggregations
   - Single interaction: need fast lookup + early filtering + different aggregation semantics

---

## Approach 1: CLI-Based Pre-Filtering (Early Filtering Strategy)

### Design Overview

Add a new CLI flag for the decision analyzer:

```bash
pdstools decision_analyzer --data-path data.parquet --interaction-id pxIxn_12345
# or multiple
pdstools decision_analyzer --data-path data.parquet --interaction-ids pxIxn_12345,pxIxn_12346
```

### Implementation Steps

1. **Extend CLI** (`python/pdstools/cli.py`)
   - Add `--interaction-id` and `--interaction-ids` options
   - Pass as environment variables: `PDSTOOLS_INTERACTION_ID`, `PDSTOOLS_INTERACTION_IDS`

2. **Modify Data Loading** (`decision_analyzer/data_read_utils.py`)
   - After `read_data()` returns LazyFrame, check for interaction filter
   - Apply `.filter(pl.col("pxInteractionID").is_in(interaction_ids))` before returning
   - Materialize the filtered result (guaranteed small) or keep lazy for compatibility

3. **Update UI** (`da_streamlit_utils.py`)
   - Add notification in Home page if interaction filter is active
   - Show which interactions are loaded
   - Optional: add UI to adjust filter (dangerous for huge datasets; disable if count > 1000)

### Pros

✓ **Deterministic & Fast**: Filter happens at load time, before any aggregations
✓ **Memory-Safe**: Guaranteed small dataset, safe to materialize if needed
✓ **Familiar Pattern**: Extends existing `--sample` and `--data-path` flags
✓ **No Architecture Changes**: Works with current lazy evaluation+caching

### Cons

✗ **User Must Know IDs**: Requires users to specify exact interaction IDs upfront
✗ **Discovery Problem**: No way to list available interactions from gigabyte files
✗ **Limited Analysis Impact**: Still leaves question of which analyses remain meaningful
✗ **Offline Workflow**: Can't filter interactively after load (would need Streamlit cache busting)

---

## Approach 2: Lazy Filtering + UI Selection (Interactive Strategy)

### Design Overview

Pass data through lazily; add an interaction selector in the UI that filters post-load:

```python
# In Home.py or a new filtering page
if st.session_state.decision_data:
    with st.expander("Advanced Filters"):
        interaction_filter_mode = st.radio(
            "Filter by interaction:",
            ["All", "Single", "Range (up to 100)"]
        )
        if interaction_filter_mode == "Single":
            # PROBLEM: How to display 10M interaction IDs?
            # Option A: Searchable dropdown (1-10s of MB in-memory)
            # Option B: Text input with validation (no upfront cost)
            interaction_id = st.text_input("Interaction ID", help="Exact match")
```

### Implementation Steps

1. **Add Filter UI** (`app/decision_analyzer/pages/0_Global_Data_Filters.py`)
   - New section: "Filter by Interaction"
   - Option 1: Text input + search (lazy validation)
   - Option 2: Multiselect with sample of IDs (show 100-1000 random sample)

2. **Update DecisionAnalyzer**
   - Accept optional `interaction_filter: list[str]` in init or as method
   - Apply filter lazily: `self.decision_data = self.decision_data.filter(...)`
   - Cache invalidation: Clear aggregation caches when filter changes

3. **Enable/Disable Analyses**
   - Analyses like Sensitivity require >= 100 interactions to be meaningful
   - Add validation in each page: `if interaction_count < threshold: st.warning("...")`

### Pros

✓ **Interactive Discovery**: Users can explore and filter without CLI knowledge
✓ **No Pre-Ingestion Cost**: All data streamed lazily; only selected slice materialized
✓ **Gradual Filtering**: Start with all, then filter down

### Cons

✗ **Discovery at Scale is Hard**: Listing 10M IDs is impractical; searching requires indexed data
✗ **Cache Invalidation**: Current caching assumes a single global dataset; filtering mid-session breaks assumptions
✗ **User Confusion**: Easy to accidentally load "all data" if filtering isn't obvious
✗ **Streamlit Complexity**: Session state management for dynamic filtering is fragile

---

## Approach 3: Hybrid (Recommended)

### Design Overview

Combine the strengths of both:

1. **CLI-level sampling/filtering** for initial data reduction (deterministic, efficient)
2. **UI filtering** for secondary exploration (interactive, discoverable)
3. **Analysis-aware constraints** to guide which pages are meaningful

### Implementation Architecture

```
CLI Entry
  ↓
  ├─ --data-path: file/directory
  ├─ --sample: optional pre-ingestion sampling (e.g., "100k")
  ├─ --interaction-id: optional single-interaction (NEW)
  ├─ --interaction-file: optional file with list of IDs (NEW)
  └─ --temp-dir: scratch (existing)
       ↓
  Data Load (read_data + filter)
       ↓
  Streamlit Home.py
       ├─ Show: data shape, interaction count, filter status
       ├─ If single interaction: show prominent notice, suggest analyses
       └─ New: "Import from file" option to add more interactions
            ↓
  Global_Data_Filters Page
       ├─ Existing filters: Stage, Channel, Direction
       ├─ NEW: Interaction ID multi-select (with search, max N results)
       └─ Apply filters lazily to decision_data
            ↓
  Analysis Pages 1-10
       ├─ Check: min_required_interactions()
       ├─ If below threshold: show info message + dimmed page
       └─ Otherwise: render analysis (same as today)
```

### Detailed Implementation

#### 1. CLI Enhancement (`cli.py`)

```python
@app.command()
def decision_analyzer(
    data_path: str = None,
    sample: str = None,
    interaction_id: str = None,           # NEW: Single interaction UUID
    interaction_file: str = None,         # NEW: File with list of IDs (one per line)
    interaction_ids: str = None,          # NEW: Comma-separated IDs
    temp_dir: str = None,
):
    """Run Decision Analyzer."""
    # Validation: mutually exclusive filters
    filter_count = sum([
        interaction_id is not None,
        interaction_file is not None,
        interaction_ids is not None,
    ])
    if filter_count > 1:
        raise ValueError("Specify only one of --interaction-id, --interaction-file, or --interaction-ids")

    # Convert to env vars
    if interaction_id:
        os.environ["PDSTOOLS_INTERACTION_ID"] = interaction_id
    if interaction_file:
        with open(interaction_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        os.environ["PDSTOOLS_INTERACTION_IDS"] = ",".join(ids)
    if interaction_ids:
        os.environ["PDSTOOLS_INTERACTION_IDS"] = interaction_ids
```

#### 2. Data Loading Enhancement (`data_read_utils.py`)

```python
def read_data(path: str, sample_spec: str = None) -> pl.LazyFrame:
    """Load data, optionally filtered by interaction."""
    data = _read_data_impl(path)  # existing multi-format logic

    # Apply pre-load filtering
    interaction_ids = get_interaction_filter()  # NEW helper
    if interaction_ids:
        data = data.filter(pl.col("pxInteractionID").is_in(interaction_ids))
        logger.debug(f"Filtered to {len(interaction_ids)} interaction(s)")

    # Existing sample logic
    if sample_spec:
        data = sample_interactions(data, sample_spec)

    return data


def get_interaction_filter() -> list[str] | None:
    """Get interaction filter from env; handle file loading if needed."""
    single_id = os.environ.get("PDSTOOLS_INTERACTION_ID")
    if single_id:
        return [single_id]

    ids_str = os.environ.get("PDSTOOLS_INTERACTION_IDS")
    if ids_str:
        return ids_str.split(",")

    return None
```

#### 3. DecisionAnalyzer Enhancement

Add method to check minimum viable analysis scope:

```python
class DecisionAnalyzer:
    def interaction_count(self) -> int:
        """Return count of unique interactions in current filtered data."""
        return self.decision_data.select("pxInteractionID").n_unique().collect()[0, 0]

    def is_valid_for_analysis(self, analysis_name: str) -> tuple[bool, str]:
        """Check if current data is suitable for a given analysis.

        Returns (is_valid, message).
        """
        count = self.interaction_count()

        # Define minimum thresholds per analysis
        THRESHOLDS = {
            "overview": 1,
            "distribution": 10,
            "funnel": 20,
            "sensitivity": 100,
            "win_loss": 50,
            "optionality": 10,
            "offer_quality": 20,
            "thresholding": 50,
            "arbitration": 5,
        }

        min_count = THRESHOLDS.get(analysis_name, 1)
        if count < min_count:
            return False, f"Requires ≥{min_count} interactions (current: {count})"
        return True, ""
```

#### 4. UI Updates: Home Page (`Home.py`)

```python
standard_page_config(page_title="Decision Analyzer")

if st.session_state.decision_data:
    # Show summary
    interaction_count = st.session_state.decision_data.interaction_count()
    action_count = st.session_state.decision_data.decision_data.shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Interactions", interaction_count)
    col2.metric("Actions", action_count)
    col3.metric("Avg Actions/Interaction", round(action_count / interaction_count, 1))

    # NEW: Show filter status if active
    if has_interaction_filter():
        st.info(f"🔍 Filtered by interaction(s): {get_filter_description()}")
        if interaction_count == 1:
            st.warning(
                "⚠️ **Single-interaction mode**: Some analyses (Sensitivity, Win-Loss) "
                "require multiple interactions to be meaningful. They will show approximate "
                "or placeholder insights."
            )

    # Suggest which pages are relevant
    st.markdown("## Suggested Next Steps")
    analyses = [
        ("1. Action Distribution", "Works well with all dataset sizes"),
        ("2. Action Funnel", "Shows pipeline flow" if interaction_count >= 20 else "Data too small; aggregate view only"),
        ("5. Global Sensitivity", "Requires ≥100 interactions" if interaction_count < 100 else "Ready to explore"),
        # ... etc
    ]
```

#### 5. UI Updates: Global Filters Page

Add Interaction ID selector (with smart discovery):

```python
def render_interaction_filter():
    """Interaction ID multi-select with search."""
    st.markdown("### Interaction ID Filter")

    max_displayed = 1000  # Limit to avoid UI lag
    unique_ids = (
        st.session_state.decision_data.decision_data
        .select("pxInteractionID")
        .unique()
        .collect()
        .to_series()
        .to_list()
    )

    if len(unique_ids) > max_displayed:
        st.caption(
            f"⚠️ {len(unique_ids):,} unique interactions. "
            f"Showing first {max_displayed}; use text search to filter."
        )
        unique_ids = unique_ids[:max_displayed]

    selected = st.multiselect(
        "Select interactions to analyze",
        unique_ids,
        help="Leave empty to analyze all interactions",
    )

    if selected:
        st.session_state.decision_data.apply_interaction_filter(selected)
        st.rerun()
```

#### 6. Analysis Pages: Validation Wrapper

Each page (e.g., `5_Global_Sensitivity_Analysis.py`) starts with:

```python
standard_page_config(page_title="Global Sensitivity Analysis · Decision Analyzer")

"# Global Sensitivity Analysis"

ensure_data()

# NEW: Check if analysis is valid for current data scope
is_valid, msg = st.session_state.decision_data.is_valid_for_analysis("sensitivity")
if not is_valid:
    st.warning(msg)
    st.info(
        "This analysis requires statistical patterns across multiple interactions. "
        "Current filtered dataset is too small to run ranking sensitivity analysis. "
        "Consider loading more data."
    )
    st.stop()

# Proceed with existing analysis code...
```

---

## Impact Analysis by Page

### Overview (Page 2) ✓ Minimal Impact

**Current**: Summary metrics (total actions, offers, avg propensity)

**Single-Interaction Impact**:
- Metrics become trivial (e.g., "Total Actions: 3", "Win Rate: 100%" if one won)
- Still meaningful: shows the actual decision made and actions considered

**Recommendation**: Keep as-is; add context callout explaining that single interactions show exact values, not statistical properties.

---

### Action Distribution (Page 3) ✓ Moderate Impact (Acceptable)

**Current**: Distribution histogram of propensities/priorities by dimension (Channel, Stage, etc.)

**Single-Interaction Impact**:
- Distribution = single bar (or small count < 10 if few interactions)
- Not statistically meaningful (can't compute percentiles, quantiles)
- But still useful: shows all actions in the decision and their propensities

**Recommendation**:
- Keep functionality but modify viz:
  - Single interaction: show table instead of binned histogram
  - < 20 interactions: show unbinned scatterplot
  - >= 20: existing binned histogram
- Add caption: "With small datasets, distributions show exact values rather than statistical patterns."

---

### Action Funnel (Page 4) ⚠️ Significant Change Needed

**Current**: Sankey flow showing how actions progress through stages (v2) or ranks (v1)

**Single-Interaction Impact**:
- Single interaction's funnel is trivial: 1 action in stage 1 → rank X → final result
- Not interesting aggregated with 1-3 interactions
- Loses value proposition (understanding systematic filtering/arbitration)

**Recommendation**:
- **Single interaction (<5)**: Switch to **Detail View** instead of Sankey
  - Show all actions in decision
  - Highlight which was selected
  - Show filtering reason (if any)
- **5 to 50 interactions**: Show simplified Sankey (fewer stages/nodes)
- **>= 50**: Show full Sankey as today

**Code Impact**: `4_Action_Funnel_Analysis.py` needs conditional logic

```python
interaction_count = st.session_state.decision_data.interaction_count()

if interaction_count < 5:
    st.info("Funnel visualization is most informative with ≥5 interactions. "
            "Showing individual decision details instead.")
    # NEW: render_single_interaction_details(data)
else:
    # Existing Sankey code
```

---

### Global Sensitivity Analysis (Page 5) ✗ Not Viable

**Current**: What-if analysis—vary ranking thresholds, show impact on selection

**Single-Interaction Impact**:
- Single interaction = single decision point
- Sensitivity analysis requires aggregate impact across many interactions
- Variance in outcome is zero

**Recommendation**:
- **Disable for < 50 interactions**
- Show explanation: "Sensitivity analysis requires ≥50 interactions to compute aggregate impact of threshold changes."
- Optional fallback: "Single-interaction hypothesis mode"—show how the action would rank under different hypothetical thresholds
  - This is **not** the same as sensitivity (different semantics)
  - Clear disclaimer: "This shows rank position change for individual action, not aggregate impact."

```python
interaction_count = st.session_state.decision_data.interaction_count()

if interaction_count < 50:
    st.error(f"Sensitivity analysis requires ≥50 interactions (current: {interaction_count})")
    st.info(
        "This analysis computes aggregate impact of threshold changes. "
        "With insufficient interactions, results are not statistically meaningful."
    )
    st.stop()
```

---

### Win-Loss Analysis (Page 6) ⚠️ Limited Usefulness

**Current**: Characteristics of selected vs. unused actions (dimensions, propensities, etc.)

**Single-Interaction Impact**:
- Single interaction = 1 selected, N unused
- Comparison is trivial (1 vs. N, not distribution vs. distribution)
- Some value: understand what made the winner stand out

**Recommendation**:
- **Single to 5 interactions**: Show as **Detail View**
  - Table: all actions, highlight winner
  - Columns: propensity, priority, stage, was_selected
- **5 to 50 interactions**: Show aggregate but with caveat (small N)
- **>= 50**: Full statistical analysis as today

```python
interaction_count = st.session_state.decision_data.interaction_count()

if interaction_count < 5:
    st.info("Win-loss analysis available in small samples as detail view (see below).")
    # render_tiny_win_loss_table(data)
elif interaction_count < 50:
    st.warning("Small sample (< 50 interactions); win-loss patterns may not be statistically reliable.")
    # render_existing_analysis_with_caveat()
else:
    # render_existing_full_analysis()
```

---

### Optionality Analysis (Page 7) ✓ Acceptable Impact

**Current**: Distribution of "how many offers shown per interaction"

**Single-Interaction Impact**:
- Single interaction = single value
- Still meaningful: shows how many options were available in that decision

**Recommendation**: Keep as-is; UI can show exact values for small N.

---

### Offer Quality Analysis (Page 8) ⚠️ Moderate Impact

**Current**: Diversity of prioritized actions, propensity spread, etc.

**Single-Interaction Impact**:
- Single interaction = single propensity spread, single diversity index
- Not a distribution; can't compute percentiles

**Recommendation**:
- Show exact values for small N
- Add callout: "Offer quality metrics become statistically meaningful with ≥20 interactions."

---

### Thresholding Analysis (Page 9) ✗ Limited Viability

**Current**: Impact of applying propensity thresholds on offer acceptance

**Single-Interaction Impact**:
- Single interaction doesn't provide statistical basis for threshold optimization
- No meaningful "before/after" distribution

**Recommendation**:
- **Disable for < 50 interactions**
- Show explanation similar to Sensitivity

---

### Arbitration Distribution (Page 10) ⚠️ Depends on Record Type

**Current**: Deep-dive into ranking logic, mandatory action patterns

**Single-Interaction Impact**:
- Depends on how many actions in the decision's arbitration stage
- If single action arbitrated: trivial
- If 5+ actions in arbitration: somewhat meaningful

**Recommendation**:
- **< 3 actions in arbitration**: Show detail view
- **3 to 20**: Show histograms with caveat (small N)
- **>= 20**: Show full analysis

---

## Summary Table: Analysis Viability by Data Size

| Analysis | 1 | 5 | 20 | 50 | 100+ |
|----------|---|---|----|----|------|
| Overview | ✓ Detail | ✓ Detail | ✓ Chart | ✓ Chart | ✓ Full |
| Distribution | ⚠️ Table | ⚠️ Scatter | ✓ Histogram | ✓ Full | ✓ Full |
| Funnel | ⚠️ Detail | ⚠️ Simple | ✓ Sankey | ✓ Full | ✓ Full |
| **Sensitivity** | ✗ | ✗ | ✗ | ⚠️ Limited | ✓ Full |
| Win-Loss | ⚠️ Detail | ⚠️ Detail | ✓ Limited | ✓ Full | ✓ Full |
| Optionality | ✓ Exact | ✓ Exact | ✓ Chart | ✓ Chart | ✓ Chart |
| Offer Quality | ⚠️ Exact | ⚠️ Exact | ✓ Chart | ✓ Full | ✓ Full |
| Thresholding | ✗ | ✗ | ✗ | ⚠️ Limited | ✓ Full |
| Arbitration | ⚠️* | ⚠️* | ✓* | ✓ Full | ✓ Full |

**Legend**: ✓ = Fully viable | ⚠️ = Reduced but useful | ✗ = Disabled / Not recommended | * = Depends on actions in arbitration stage

---

## User Communication Strategy

### Home Page Alert

When interaction filter is active:

```markdown
🔍 **Filtered Mode Active**
- Interactions: 3 (out of 1,234,567 total)
- Analysis pages will show behavior for these specific decisions

⚠️ **Note**: Some analyses require statistically diverse datasets:
  - ❌ Sensitivity Analysis: Disabled (needs ≥50 interactions)
  - ⚠️ Win-Loss Analysis: Shows individual comparisons (not aggregate patterns)
  - ✓ Action Distribution: Shows actual actions in each decision

[Learn more about filtering](#)
```

### Per-Page Messaging

Each analysis page that's impacted should lead with a context callout:

```
**Single-Interaction Mode**: This analysis shows [exact values / approximate patterns]
based on your current 1 decision. [Learn how page changes with more data](#).
```

---

## Implementation Roadmap (Phased Approach)

### Phase 1: CLI Filtering (Foundation)
- Add `--interaction-id` / `--interaction-ids` / `--interaction-file` flags
- Update data loading to apply filter pre-ingestion
- Show filtered status in Home page
- Estimated effort: **2-3 days**

### Phase 2: Analysis Validation (Safety)
- Add `is_valid_for_analysis()` method to DecisionAnalyzer
- Add `st.stop()` guards to pages that require N interactions
- Test each page manually with 1, 5, 20, 50 interactions
- Estimated effort: **3-4 days**

### Phase 3: Graceful Degradation (UX)
- Modify Summary/Overview page to show interactive details for N<10
- Modify Distribution page to show scatterplot for N<20
- Modify Funnel to show detail view for N<5
- Estimated effort: **4-5 days**

### Phase 4: UI Filtering (Polish)
- Add Interaction ID multiselect to Global Filters page
- Cache invalidation on filter change
- Performance testing with large interaction lists
- Estimated effort: **3-4 days**

### Phase 5: Documentation & Examples
- Document filtering modes, suitable use cases
- Add examples to docs (single-interaction deep-dive workflow)
- Create tutorial notebook
- Estimated effort: **2-3 days**

**Total estimate**: **14-19 days** for full implementation

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Users misinterpret single-interaction analysis as statistical pattern | High | Clear messaging on every affected page; disable problematic analyses |
| Performance: filtering large datasets (10M interactions) in-memory | High | Keep all filtering lazy; only materialize selected interactions |
| Cache invalidation logic becomes complex with dynamic filtering | Medium | Document assumptions; test session state edge cases |
| Multiple filter dimensions compete (sample + interaction ID) | Medium | Validate CLI inputs; make mutual exclusivity explicit |
| UI becomes cluttered with conditional content | Medium | Use clear sections; progressive disclosure (collapsible warnings) |

---

## Recommended Design Decision

### Hybrid Approach (CLI + UI + Safe Analysis Boundaries)

**Rationale:**
1. **CLI filtering solves the discovery problem** without expensive in-memory operations
2. **UI filtering enables interactive exploration** post-load
3. **Analysis validation prevents misinterpretation** of invalid analyses
4. **Graceful degradation preserves usability** at all data scales

**Key principles:**
- Start filtering early (CLI) to keep data small
- Communicate constraints clearly (Home page, per-page callouts)
- Disable analyses that are statistically invalid (Sensitivity, Thresholding < 50 interactions)
- Degrade gracefully for marginal cases (show detail instead of histogram)
- Test extensively with small datasets to catch UI/cache issues

---

## Questions for Product/UX

1. **Primary Use Case**: Are users debugging a single bad decision? Auditing a customer interaction? Exploring patterns in a segment?
   - Answer determines whether single-interaction drill-down is exploratory or post-incident investigation.

2. **Interaction List Discoverability**: Should we support importing interaction IDs from:
   - CSV file uploaded by user?
   - REST API query (e.g., "last 10 decisions for customer X")?
   - Manual text input (safest for small N)?

3. **Sensitivity/Thresholding Fallback**: For disabled analyses, should we offer:
   - Disabled page (stop early)?
   - Hypothesis mode (show what-if for the individual interaction)?
   - Omit from sidebar entirely?

4. **Backwards Compatibility**: Should existing CLI/UI remain unchanged?
   - Recommendation: Yes. Add new flags; don't alter existing `--sample` behavior.

---

## References

- Current CLI: `python/pdstools/cli.py`
- Data loading: `python/pdstools/decision_analyzer/data_read_utils.py`
- DecisionAnalyzer class: `python/pdstools/decision_analyzer/DecisionAnalyzer.py`
- Analysis pages: `python/pdstools/app/decision_analyzer/pages/`
- Streamlit utilities: `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`

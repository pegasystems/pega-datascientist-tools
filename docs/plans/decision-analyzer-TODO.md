# Decision Analyzer — Backlog

Active work items for the Decision Analysis Tool.

---

## Core Library

### High Priority

- [ ] **Fix `num_samples > 1`** — Pre-aggregation sampling ([DecisionAnalyzer.py:569](../python/pdstools/decision_analyzer/DecisionAnalyzer.py#L569)) is locked at 1 because >1 breaks thresholding. **Issue:** With multiple samples per group, `.explode()` in `getThresholdingData` ([line 1236](../python/pdstools/decision_analyzer/DecisionAnalyzer.py#L1236)) creates multiple rows per aggregated group, biasing the quantile calculation toward groups with more underlying interactions. **Solutions:** (A) Weight samples properly in thresholding, (B) Calculate quantiles at group level before exploding, (C) Use min/max columns instead of samples for distributions, or (D) Document the limitation and keep `num_samples=1`.

### Medium Priority

- [ ] **Raw IH data support** — Test how well Decision Analyzer works with raw Interaction History data. IH is similar to EE v1 but may lack a few columns; identify gaps and decide whether to support it as a first-class input format or document limitations.
- [ ] **Refactor `get_offer_quality`** — Uses a manual stage loop; should delegate to `aggregate_remaining_per_stage`.
- [ ] **Win rank flexibility** — `get_win_loss_distribution_data` has a fixed rank parameter. Return all ranks, let UI filter. Make `max_value` data-driven (not hardcoded 10).
- [ ] **Handle Mandatory Actions** — Mandatory actions use a special arbitration priority (4999999 or possibly 4999999999 in data) that bypasses normal prioritization. This affects sensitivity analysis, win/loss distributions, and component distributions. See [Pega docs: NBA Strategy Arbitration](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/nba-strategy-arbitration.html). **Existing support:** `DecisionAnalyzer` already accepts a `mandatory_expr` parameter (a polars expression) to tag issue/group/action combinations as mandatory at init time, and ranking sorts by the resulting `is_mandatory` field before Priority. **Remaining work:** auto-detect mandatory actions from the priority value in the data (instead of requiring an explicit expression), flag/annotate them in the UI, and exclude them from sensitivity analyses (their fixed high priority makes them insensitive to lever changes, skewing results). Also consider excluding or annotating them in component distributions and win/loss analyses.
- [ ] **Distinct propensity display names** — `column_schema.py` has model propensity and propensity sharing the same display name. Give them distinct names (e.g. "Model Propensity" vs "Propensity") and update PVCL code.

### Low Priority

- [ ] **Pre-aggregated data visualization** — The Infinity Action Analysis feature exports  pre-computed results (sensitivity, funnel data) rather than raw interactions. Could bypass `DecisionAnalyzer` and visualize directly.
- [ ] **Add treatment to scope hierarchy** — Currently commented out in `ExplainabilityExtract` schema. Requires design decisions about aggregation strategy and user choice. See detailed analysis in "Treatment Support Analysis" section below.
- [ ] **Customer-level aggregates** — Per-customer stats (postponed; current data not representative).
- [ ] **AB test: include IA properties** — `getABTestResults` should include Impact Analyzer properties when populated.
- [ ] **Optimize thresholding quantiles** — Current implementation is verbose and potentially slow.
- [ ] **Stratified sampling option** — Current `sample` property is random by interaction ID. Consider optional stratification by channel/direction.
- [ ] **Scale up counts in UI after sampling** — Future enhancement: multiply displayed counts by inverse of sample fraction to show estimated real volumes (metadata infrastructure now in place).
- [ ] **Streaming pre-aggregation** — `getPreaggregatedFilterView` calls `.collect()` on the full dataset. Investigate polars streaming or chunked processing for GB-scale data.

---

## Streamlit App — Architecture

- [ ] **Dynamic stage UI** — Mostly data-driven, but some hardcoded stages remain: [8_Optionality_Analysis.py:97-100](../python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py#L97-L100) hardcodes `stage="Output"` for action variation (should use last stage from data or make selectable); [plots.py:439](../python/pdstools/decision_analyzer/plots.py#L439) filters out `"Final"` stage without checking if it exists.
- [ ] **HC data import alignment** — Health Check still uses its own `import_datamart()` pattern with different labels. Could be aligned.
- [ ] **Promote `data_read_utils` to pdstools core** — The data-reading facilities in `decision_analyzer/data_read_utils.py` (multi-format ingestion, schema detection, lazy scanning) are generic enough to serve all apps and potentially replace the legacy `readDSExport` family. Evaluate extracting them into `pdstools/pega_io` or a top-level utility module.

---

## Streamlit Pages

### 2 — Global Data Filters
- [ ] Robustness against heavy filtering (e.g. dropping stages)
- [ ] Apply filters by default when set

### 3 — Overview
- [ ] Speed up first-use (too much computation upfront)
- [ ] Add Value Finder-style pie chart at arbitration

### 4 — Action Distribution
- [ ] Dynamic stages from data
- [ ] Top-K limiter for bar charts; show all ticks or indicate truncation
- [ ] Multi-stage selection

### 5 — Action Funnel
- [ ] Toggle between Passing and Filtered perspective
- [ ] Better coloring for filter components (by type)
- [ ] Move top-N control from sidebar to section
- [ ] Component overlap analysis, component impact over time (trend)

### 6 — Global Sensitivity
- [ ] Infer top-X from data (max rank per channel for Final records)
- [ ] Win rank upper bound from data, not hardcoded

### 7 — Win/Loss Analysis
- [ ] Verify numbers (bar charts vs box plots)
- [ ] Consistent colors across paired charts; fix pale colors for small counts
- [ ] Handle many channels gracefully
- [ ] Generalize rank usage across pages

### 8 — Optionality Analysis
- [ ] Overlay propensity + priority on optionality plot
- [ ] Consistent stage color scheme across all plots

### 9 — Offer Quality Analysis
- [ ] Generalize stage naming, session state cleanup, move logic to class

### 10 — Thresholding Analysis
- [ ] **Move inline plots to `plots` module** — Three plotly charts ([lines 134-174, 202-209](../python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py#L134-L174)) defined inline. Extract to reusable functions in `plots.py`.
- [ ] **Depends on `num_samples = 1`** — Page uses `.explode()` throughout (lines 31-32, 155, 187-188), so it breaks if `num_samples > 1` is ever enabled. See High Priority item above.

### 11 — Arbitration Component Distribution
- [ ] **EE v1: include all propensity-like properties** — Explainability Extract (v1) has multiple propensity flavors (e.g. model propensity, adjusted propensity) whereas v2 only has one. The component overview and detail views should surface all propensity-like columns for v1 data.
- [ ] Finish proposition distribution side-by-side view

### 11 — Business Lever Analysis *(hidden — needs full rework)*

> **Status:** Page moved to `_stashed/Business_Lever_Analysis.py` as of the `feat/external-user-readiness` branch. Not ready for external users. Before re-enabling, the page needs a thorough redesign.
>
> **Purpose:** Support what-if analysis of lever changes — let users simulate the effect of adjusting business levers on win distributions and, ultimately, automatically find the right lever values to reach specific goals (target win rates, volume targets, etc.).
>
> **Why hidden:** The UX is confusing (unclear terminology, non-intuitive controls), the code bypasses the standard data/session-state patterns used by other pages, and the lever-finder feature is too rough for production use.
>
> **To re-enable:** Address the items below, rename the file back to a numbered page (e.g. `11_Business_Lever_Analysis.py`), and renumber About accordingly.

- [ ] Redesign UX: clearer terminology, intuitive controls, side-by-side before/after distributions
- [ ] Generalize to work across a selection of actions, not just a single action
- [ ] Refactor lever calculation into `DecisionAnalyzer` class
- [ ] Use aggregated/sampled data (not raw `decision_data`)
- [ ] Move plots to `plots` module; clean up session state
- [ ] Start target win ratio at > 0
- [ ] Share distribution visualization code with Thresholding Analysis page
- [ ] Improve lever-finder algorithm robustness and UX

---

## Plots (`plots.py`)

- [ ] **Plotly box sizing** — Multiple box plots ignore size constraints. Adopt the solution from ADM Datamart Plots.
- [ ] **Legend suppression** — `showlegend=False` not working in some polar/bar charts.
- [ ] **Hover info** — Add hover showing individual colored totals and total bar in stacked histograms.
- [ ] **Pie chart stage limit** — Temporary cap at 5 stages; make dynamic.

---

## Data Size & Performance

- [ ] **Data size warning in UI** — Show warning when data exceeds a practical threshold (>500 MB, >5M rows).

---

## Treatment Support Analysis

**Status:** Under investigation. Treatment column is currently commented out in `ExplainabilityExtract` schema (column_schema.py:202). Before enabling, need to resolve aggregation strategy and user experience questions.

### Background

EE v2 exports contain a granularity hierarchy: **Issue → Group → Action → Treatment**

- Not all customers use treatments (~0.6% of export rows in sample data)
- When present: ~3 treatments per action on average (range: 1-16)
- Current code assumes action-level granularity: `scope_hierarchy = ["Issue", "Group", "Action"]` (DecisionAnalyzer.py:907)

### The Core Problem

When treatment data exists, **the same action appears multiple times per interaction** with different treatments. Each treatment has its own propensity and priority values. This creates duplication when analyzing at the action level.

**Example from sample data:**
- Action: "SavingsNudgeMoneyThor" in single interaction appears twice:
  - Treatment A: propensity 0.979, priority 0.979
  - Treatment B: propensity 0.564, priority 0.564

For action-level analysis (e.g., "which actions win at arbitration?"), we need to either:
1. **Aggregate treatments to actions** — collapse multiple treatment rows into one action row
2. **Support treatment-level granularity** — extend all analyses to work at treatment level

### Data Behavior (from 81M row sample)

**Treatment prevalence:**
- 470K rows (0.6%) have treatment values; 81M (99.4%) do not
- 284 actions have treatments; 820 unique treatments total
- At Output (Final) stage: 235 actions appear with multiple treatments (~0.4% duplication)

**Treatment scoring:**
- Treatments appear AFTER action-level propensity scoring
- Pipeline stages: TreatmentJoins (2600) → TreatmentAvailability (2900) → TreatmentPlacements (4900)
- Each treatment gets its own `FinalPropensity` and `Priority` (Priority = FinalPropensity)
- For same action, propensities vary significantly by treatment:
  - Example: "OnlineStatements" (16 treatments) — propensities range 0.41 to 0.59
  - Each treatment represents a different creative/channel variant

**Arbitration implications:**
- Multiple treatments for same action compete in arbitration separately
- Each treatment-action pair has its own rank and win/loss record
- When aggregating to action level, unclear which treatment's metrics to use

### Open Questions

#### 1. Aggregation Strategy for Action-Level Analysis

When rolling up treatments to actions, which treatment "represents" the action?

**Option A: Max Propensity**
- Use treatment with highest `FinalPropensity`
- Rationale: Most likely to convert
- Issue: May not be the one selected by arbitration if other factors matter

**Option B: Max Priority**
- Use treatment with highest `Priority`
- Rationale: Most likely to rank highly
- Issue: Priority = Propensity in this data, so equivalent to Option A

**Option C: Best Rank**
- Use treatment with best (lowest) rank after arbitration
- Rationale: Reflects actual selection logic
- Issue: Requires computing rank first; rank is stage-specific

**Option D: Weighted Average**
- Aggregate propensity/priority by some weighting (e.g., by treatment occurrence frequency)
- Rationale: Represents "typical" treatment performance
- Issue: Unclear what weight to use; loses information about best-case

#### 2. Aggregation Scope

Which metrics need aggregation, and how?

| Metric | Current Type | Aggregation Strategy |
|--------|--------------|---------------------|
| `FinalPropensity` | Float | Max? Mean? Take from selected treatment? |
| `Priority` | Float | Max (makes sense for ranking) |
| `Value` | Float | Max? Mean? Take from selected treatment? |
| `ContextWeight` | Float | Same as Value? |
| `Weight` (Levers) | Float | Same as Value? |
| `Rank` | Int | Min (best rank) |

#### 3. User Experience

Should treatment support be:

**Option A: Automatic Aggregation**
- Always aggregate treatments to actions at data load
- Pro: Simpler UX; action-level view matches user mental model
- Con: Loses treatment-level detail; user can't analyze treatment performance

**Option B: User Choice**
- Add UI toggle: "Action-level" vs "Treatment-level" analysis
- Pro: Preserves full data; users who care about treatments can analyze them
- Con: More complex UX; need to handle both modes throughout app

**Option C: Auto-detect + Explicit Choice**
- If treatments present, prompt user at data load: "Aggregate to actions or analyze treatments?"
- Store choice in `DecisionAnalyzer` instance
- Pro: Explicit about the decision; users understand what they're analyzing
- Con: Extra interaction; may confuse users unfamiliar with treatments

### Recommended Investigation Steps

Before implementing, need to:

1. **Validate arbitration behavior**
   - When multiple treatments for same action compete, which wins?
   - Does the highest-ranked treatment always have highest propensity?
   - Test: For actions with multiple treatments, check if rank correlates with propensity

2. **Understand treatment selection in Pega**
   - Is treatment selection deterministic (same action always picks same treatment)?
   - Or does it vary by context (same action can pick different treatments for different customers)?
   - Check Pega docs on treatment arbitration logic

3. **Test aggregation strategies**
   - Implement both "max propensity" and "best rank" aggregation
   - Compare results against raw treatment-level data
   - Validate that action-level win/loss metrics make sense

4. **Check v1 vs v2 differences**
   - Does EE v1 have treatments? (Need to test with v1 data)
   - Do v1 and v2 handle treatments differently?

5. **User research**
   - Do users expect to see treatments in the tool, or just actions?
   - What questions do users have about treatment performance?
   - Would treatment-level analysis add value, or just complexity?

### Implementation Considerations

**If aggregating at data load:**
- Add treatment aggregation step in `cleanup_raw_data()` when Treatment column is present
- Use max propensity or best rank to select representative treatment
- Drop Treatment column after aggregation so rest of code works unchanged

**If supporting both granularities:**
- Add `granularity` parameter to `DecisionAnalyzer.__init__`: `"action"` (default) or `"treatment"`
- Extend `scope_hierarchy` to optionally include `"Treatment"`: `["Issue", "Group", "Action", "Treatment"]`
- Update all groupby operations: `get_win_distribution_data`, `aggregate_remaining_per_stage`, `get_component_impact_detailed`, etc.
- Add UI control to switch between action/treatment views
- Handle case where treatment data doesn't exist (fall back to action level)

**Testing needs:**
- Test with data containing treatments (sample_99k_1.parquet)
- Test with data without treatments (need to find sample)
- Test with v1 data (if v1 supports treatments)
- Verify win/loss, sensitivity, and ranking metrics are correct at both granularities

### Related TODO Items

This work relates to:
- [ ] **Distinct propensity display names** (line 21) — Need to clarify model propensity vs final propensity, especially for treatments
- [ ] **Handle Mandatory Actions** (line 20) — Mandatory actions may or may not have treatments; ensure aggregation doesn't break mandatory logic

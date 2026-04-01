# Decision Analyzer — Backlog

Open work items for the Decision Analysis Tool. Kept current — items are
removed or marked done as commits land on master.

**Priority levels:** P1 = high / P2 = medium / P3 = nice-to-have

---

## Core Library

### P1

- [ ] **Fix `num_samples > 1`** — Locked at 1 because `.explode()` in `get_thresholding_data` biases quantile calculations when groups have multiple samples. Options: weight samples, calculate quantiles before exploding, or document the limitation.

- [ ] **Handle mandatory actions** — Priority 4999999+ bypasses normal ranking, skewing sensitivity and win/loss analysis. `DecisionAnalyzer` already accepts `mandatory_expr` and sorts by `is_mandatory`. Remaining: auto-detect from priority value, flag in UI visualizations, exclude or separate in sensitivity analysis.

### P2

- [x] **Remove backward-compat aliases** — Removed all backward-compat aliases and updated example notebooks, app code, and tests to use the current method names.
  - ~~camelCase aliases (16 methods): `applyGlobalDataFilters`, `getDistributionData`, `getFunnelData`, etc.~~
  - ~~`winning_from` / `losing_to` aliases for the now-private `_winning_from` / `_losing_to`~~
  - ~~`get_optionality_data_with_trend` — replaced by `get_optionality_data(by_day=True)`~~
  - ~~`get_overview_stats` method — replaced by `overview_stats` cached property~~

- [ ] **Consolidate win/loss methods** — `_winning_from`/`_losing_to` duplicate logic (only the rank comparison differs). Merge into single private helper with `direction` param; review whether `get_win_loss_distribution_data` and `get_win_loss_distributions` can be unified.

- [ ] **Refactor `get_offer_quality`** — Uses manual stage loop instead of delegating to `aggregate_remaining_per_stage`. Rewrite to use the shared aggregation, verify results match, update tests.

- [ ] **Win rank flexibility** — `get_win_loss_distribution_data` has hardcoded `max_value=10`. Compute from actual max rank in data; return all ranks and let UI filter.

- [ ] **Distinct propensity display names** — `column_schema.py` maps model propensity and final propensity to the same display name. Use "Model Propensity" vs "Final Propensity".

- [ ] **Raw IH data support** — Test with raw Interaction History format (EE v1-like). Identify missing columns, decide whether to support as first-class format or document limitations.

### P3

- [ ] **Pre-aggregated data** — Detect Infinity Action Analysis pre-computed results and bypass `DecisionAnalyzer` aggregation.
- [ ] **Treatment support** — Treatment column commented out in schema. When present, same action appears multiple times per interaction with different propensities. Key decisions needed: aggregation strategy, UX approach, which metrics need treatment-specific handling. Low prevalence (~0.6% of rows).
- [ ] **Customer-level aggregates** — Per-customer statistics. Postponed until representative multi-customer dataset available.
- [ ] **AB test: include IA properties** — `get_ab_test_results` doesn't join Impact Analyzer properties when present.
- [ ] **Stratified sampling** — Add optional `sample_stratified_by` parameter for proportional representation across channels/directions.
- [ ] **Scale up counts after sampling** — Show estimated full-volume counts in UI (infrastructure in place via sample fraction metadata).
- [ ] **Streaming pre-aggregation** — `preaggregated_filter_view` calls `.collect()` on full dataset. Investigate polars streaming mode for GB-scale data.

---

## Streamlit App — Architecture

### P1

- [ ] **Custom sidebar navigation with disabled pages** — Streamlit hides unavailable pages. Define a page registry with prerequisites and availability predicates; render all pages, grey out unavailable ones with reason messages; add `ensure_page_access()` guard.

### P2

- [ ] **Improve file upload robustness** — Filter META-INF/MANIFEST.mf files, clear session state on new upload, improve error messages. Apply patterns from Impact Analyzer.
- [ ] **Dynamic stage UI** — Hardcoded stage names remain in Optionality page (`stage="Output"`) and plots.py (`"Final"` filter). Replace with data-driven `DecisionAnalyzer.stages`.

### P3

- [ ] **Promote `data_read_utils` to pdstools core** — Generic multi-format ingestion in `decision_analyzer/data_read_utils.py` could serve all apps.

---

## Streamlit Pages

### Page 2 — Global Data Filters

- [ ] **[P2] Robustness against heavy filtering** — Validate after filter application; warn if >95% of data eliminated; provide "reset filters" escape hatch.

### Page 3 — Overview

- [ ] **[P1] Speed up first-use page load** — Lazy-load stage-specific metrics per tab instead of computing everything upfront.
- [ ] **[P2] Value Finder-style pie chart** — Add win/loss proportion pie at arbitration stage for users migrating from Value Finder.

### Page 4 — Action Distribution

- [ ] **[P2] Top-K limiter** — Bar charts with >50 actions are unreadable. Add "Show top N" control with sorting options.
- [ ] **[P2] Dynamic stages** — Audit for hardcoded stage names; use `DecisionAnalyzer.stages`.

### Page 5 — Action Funnel

- [ ] **[P2] Funnel summary table improvements** — Current table shows averages/percentages. Add absolute-count display mode toggle; include synthetic "Available Actions" baseline row; clarify headers.
- [ ] **[P2] Better coloring for filter components** — Use consistent color scheme by filter category (eligibility, suitability, engagement).
- [ ] **[P3] Component overlap analysis** — UpSet plot or Sankey showing which actions are hit by multiple filters.
- [ ] **[P3] Component impact over time** — Time-series trend of filter removal rates.

### Page 6 — Global Sensitivity

- [ ] **[P2] Infer top-X and win rank bounds from data** — Auto-compute from actual max rank instead of user-specified or hardcoded values.

### Page 7 — Win/Loss Analysis

- [ ] **[P2] Verify numbers across visualizations** — Bar charts and box plots sometimes show different counts. Audit data sources.
- [ ] **[P2] Deep-dive selectors for specific opponents** — Scope deep-dive to selected counterpart items the comparison group wins to / loses from.
- [ ] **[P3] Shared rank selector component** — Standardize rank selection UI across Win/Loss, Sensitivity, and other pages.

### Page 8 — Optionality Analysis

- [ ] **[P2] Overlay propensity/priority on optionality plot** — Dual-axis or faceted view alongside optionality distribution.
- [ ] **[P2] Color optionality plot by issue/group** — Add selector to color the optionality distribution by Issue or Group. Helps explain bi-modal patterns where different issues/groups dominate different parts of the distribution.

### Page 9 — Offer Quality Analysis

- [ ] **[P2] Generalize stage naming** — Audit for hardcoded stages; use data-driven stage selection.
- [ ] **[P2] Move logic to DecisionAnalyzer** — Extract offer quality calculations from Streamlit page code.

### Page 10 — Thresholding Analysis

- [ ] **[P2] Move inline plots to `plots` module** — Three plotly charts defined inline. Extract to reusable functions.
- [ ] **[P1] Depends on `num_samples = 1`** — Uses `.explode()` on samples. Coordinate fix with core library `num_samples` item.

### Page 11 — Arbitration Component Distribution

- [ ] **[P2] EE v1: show all propensity flavors** — V1 has model/adjusted/final propensity; currently only shows one.
- [ ] **[P3] Proposition distribution side-by-side** — Complete multi-select comparison of component distributions.

### Hidden — Business Lever Analysis *(stashed, needs full rework)*

Page in `_stashed/Business_Lever_Analysis.py`. Confusing UX, bypasses data patterns, lever-finder too rough. To re-enable:

- [ ] **[P1] Redesign UX** — Clarify terminology; side-by-side before/after distributions.
- [ ] **[P1] Generalize to action sets** — Currently single-action only. Allow group selection.
- [ ] **[P1] Refactor lever calculation into DecisionAnalyzer** — Logic is embedded in page code.
- [ ] **[P2] Use aggregated/sampled data** — Currently bypasses normal data flow.
- [ ] **[P2] Move plots to `plots` module** — Inline chart definitions.

---

## Plots (`plots.py`)

- [ ] **[P2] Plotly box sizing** — Box plots ignore size constraints. Adopt explicit height/width with responsive scaling.
- [ ] **[P2] Hover info in stacked histograms** — Show both segment value and total bar height on hover.
- [ ] **[P3] Pie chart stage limit** — Hardcoded cap at 5 stages. Make dynamic; switch to bar chart for >5.

---

## Performance

- [ ] **[P2] Data size warning** — No feedback when loading large files. Warn at >500MB / >5M rows; suggest sampling.

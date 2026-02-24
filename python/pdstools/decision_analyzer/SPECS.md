# Decision Analyzer — Backlog

Tracked on branch: `refactor/decision-analyzer`

---

## Core Library

### High Priority

- [ ] **Fix `num_samples > 1`** — Pre-aggregation sampling locked at 1 because >1 breaks `.explode()` in thresholding. Either fix thresholding or document limitation.
- [ ] **Validate `fields_for_data_filtering`** — Subset against actual available columns instead of hardcoded list.
- [ ] **Human-friendly stage names** — Map internal names to display names (e.g. "Final" → "Presented"). May need a config dict. *(See also: `DecisionAnalyzer.py` TODO)*
- [ ] **Graceful degradation for minimal data** — Exports with only context keys (no scoring columns) crash with `ColumnNotFoundError`. Show a clear message about missing columns; ideally support a reduced analysis mode (action distribution only).

### Medium Priority

- [ ] **Refactor `get_offer_quality`** — Uses a manual stage loop; should delegate to `aggregate_remaining_per_stage`.
- [ ] **Win rank flexibility** — `get_win_loss_distribution_data` has a fixed rank parameter. Return all ranks, let UI filter. Make `max_value` data-driven (not hardcoded 10).
- [ ] **Caching for expensive methods** — `getThresholdingData` and `get_sensitivity` are expensive but uncached. Consider `lru_cache` or polars caching.
- [ ] **Distinct propensity display names** — `column_schema.py` has model propensity and propensity sharing the same display name. Give them distinct names (e.g. "Model Propensity" vs "Propensity") and update PVCL code.
- [x] **Level selection** — User can switch between "Stage Group" and "Stage" granularity from page sidebars (pages 3, 4, 7, 8). V1 data only exposes "Stage Group".

### Low Priority

- [ ] **Pre-aggregated data visualization** — Some exports contain pre-computed results (sensitivity, funnel data) rather than raw interactions. Could bypass `DecisionAnalyzer` and visualize directly.
- [ ] **Add treatment to scope hierarchy** — Currently commented out in `NBADScope_Mapping`.
- [ ] **Customer-level aggregates** — Per-customer stats (postponed; current data not representative).
- [ ] **AB test: include IA properties** — `getABTestResults` should include Impact Analyzer properties when populated.
- [ ] **Optimize thresholding quantiles** — Current implementation is verbose and potentially slow.
- [x] **Fix `sample` docstring** — Says "taking the first 50,000 interactions" but actually uses hash-based sampling.
- [ ] **Stratified sampling option** — Current `sample` property is random by interaction ID. Consider optional stratification by channel/direction.
- [ ] **Streaming pre-aggregation** — `getPreaggregatedFilterView` calls `.collect()` on the full dataset. Investigate polars streaming or chunked processing for GB-scale data.

---

## Streamlit App — Architecture

- [ ] **Dynamic stage UI** — Stages should be data-driven with arbitrary count. Several pages hardcode stage names.
- [ ] **Session state cleanup** — Too much stored in `st.session_state` (especially pages 8, 11). Review and minimize.
- [ ] **Move inline plot code to `plots` module** — Business lever analysis (page 11) and others have plots defined inline.
- [ ] **HC data import alignment** — Health Check still uses its own `import_datamart()` pattern with different labels. Could be aligned.

---

## Streamlit Pages

### 1 — Global Data Filters
- [ ] Customize filter dropdown (focus on logical filter fields)
- [ ] Add reset button for filters
- [ ] Robustness against heavy filtering (e.g. dropping stages)
- [ ] Apply filters by default when set

### 2 — Overview
- [ ] Speed up first-use (too much computation upfront)
- [ ] Add Value Finder-style pie chart at arbitration

### 3 — Action Distribution
- [ ] Dynamic stages from data
- [ ] Top-K limiter for bar charts; show all ticks or indicate truncation
- [ ] Multi-stage selection

### 4 — Action Funnel
- [ ] Toggle between Passing and Filtered perspective
- [ ] Better coloring for filter components (by type)
- [ ] Move top-N control from sidebar to section
- [ ] Component overlap analysis, component impact over time (trend)

### 5 — Global Sensitivity
- [ ] Infer top-X from data (max rank per channel for Final records)
- [ ] Win rank upper bound from data, not hardcoded

### 6 — Win/Loss Analysis
- [ ] Verify numbers (bar charts vs box plots)
- [ ] Consistent colors across paired charts; fix pale colors for small counts
- [ ] Handle many channels gracefully
- [ ] Generalize rank usage across pages

### 7 — Personalization Analysis
- [ ] Overlay propensity + priority on optionality plot
- [ ] Consistent stage color scheme across all plots
- [ ] Fix session state key warning

### 8 — Offer Quality *(⚠ incomplete)*
- [ ] Generalize stage naming, session state cleanup, move logic to class

### 9 — Thresholding *(⚠ incomplete)*
- [ ] Largely broken: explode issue with `num_samples > 1`, filtering, interactive thresholds
- [ ] Show volume/distribution change with grouped bar charts
- [ ] Consider rewrite or removal

### 10 — Arbitration Component Distribution
- [ ] Finish proposition distribution side-by-side view

### 11 — Business Lever Analysis
- [ ] Refactor lever calculation into `DecisionAnalyzer` class
- [ ] Use aggregated/sampled data (not raw `decision_data`)
- [ ] Move plots to `plots` module; clean up session state
- [ ] Start target win ratio at > 0

---

## Plots (`plots.py`)

- [ ] **Plotly box sizing** — Multiple box plots ignore size constraints. Adopt the solution from ADM Datamart Plots.
- [ ] **Legend suppression** — `showlegend=False` not working in some polar/bar charts.
- [ ] **Hover info** — Add hover showing individual colored totals and total bar in stacked histograms.
- [ ] **Pie chart stage limit** — Temporary cap at 5 stages; make dynamic.

---

## Data Size & Performance

- [ ] **Data size warning in UI** — Show warning when data exceeds a practical threshold (>500 MB, >5M rows).
- [x] **Update `sample` property docstring** — Clarify it operates on potentially pre-sampled data (two layers of reduction possible).

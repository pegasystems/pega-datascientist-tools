# Decision Analyzer — Refactoring Specs & Work Items

Tracked on branch: `refactor/decision-analyzer`

## Completed

- [x] **Test harness**: 89 unit tests covering v1/v2 core functionality
- [x] **File rename**: `decision_data.py` → `DecisionAnalyzer.py`
- [x] **Class method constructors**: `from_explainability_extract()`, `from_decision_analyzer()`
- [x] **Deprecated polars APIs**: `pl.count()→pl.len()`, `with_row_count→with_row_index`, `melt→unpivot`, `Categorical(ordering=)→Categorical`, `ColumnNotFoundError` import
- [x] **Graceful error handling**: `ValueError` on missing critical columns
- [x] **Version in UI**: `pdstools.__version__` replaces hardcoded `app_version`
- [x] **Dead code removal**: `get_git_version_and_date()`, unused imports
- [x] **Performance fix**: `.columns` → `.collect_schema().names()`
- [x] **Print cleanup**: `print()` → `logger.debug()` in `sample()`
- [x] **v1/v2 UI clarity**: comparison table in Home.py, format detection feedback, improved `ensure_funnel()` message

---

## Core Library

### High Priority

- [ ] **Generalize context keys** — `pyIssue/pyGroup/pyName` are hardcoded everywhere. Should be configurable or inferred from data. Affects: `getActionVariationData`, `get_first_level_stats`, `reRank`, `get_sensitivity`.
- [x] **Handle missing PVCL components** — `reRank()` now checks which columns exist and fills missing ones with 1.0 (neutral for multiplication).
- [ ] **Fix num_samples > 1** — Pre-aggregation sampling locked at 1 because >1 breaks `.explode()` in thresholding. Either fix thresholding or document limitation.
- [ ] **Validate fields_for_data_filtering** — Subset against actual available columns instead of hardcoded list.
- [ ] **Human-friendly stage names** — Map internal names to display names (e.g. "Final" → "Presented"). May need a config dict.
- [x] **Channels + direction** — `get_overview_stats` now counts unique channel/direction combinations (falls back to channel-only if pyDirection absent).

### Medium Priority

- [ ] **Move `filtered_action_counts` into class** — Standalone function in `utils.py` that duplicates the `aggregate_remaining_per_stage` pattern.
- [ ] **Refactor `get_offer_quality`** — Uses a manual stage loop; should delegate to `aggregate_remaining_per_stage`.
- [ ] **Win_rank flexibility** — `get_win_loss_distribution_data` has fixed rank parameter. Return all ranks, let UI filter.
- [ ] **Caching for parameterized methods** — `getThresholdingData` and `get_sensitivity` are expensive but not cached. Consider `lru_cache` or polars caching.
- [ ] **DRY prio factor list** — `["Propensity", "Value", "Context Weight", "Levers"]` repeated in DecisionAnalyzer.py and plots.py. Define once.

### Low Priority

- [ ] **Add treatment to NBADScope_Mapping** — Currently commented out.
- [ ] **Customer-level aggregates** — May need per-customer stats (postponed, current data not representative).
- [ ] **AB test: include IA properties** — `getABTestResults` should include Impact Analyzer properties when populated.
- [ ] **Optimize thresholding quantiles** — Current implementation is verbose and potentially slow.

---

## Streamlit App — Architecture

### Prerequisites (do before page work)

- [x] **Caching overhaul** — Added `@st.cache_resource` for `load_decision_analyzer()`. Home.py no longer nukes all state with `st.session_state.clear()`. Removed dead commented-out `@st.cache_data` wrappers and unused imports.
- [ ] **Split `get_data_filters()`** — 200-line function in `da_streamlit_utils.py` doing too much. Break into: categorical filter, numeric filter, temporal filter, multiselect manager, state cleanup.
- [ ] **Dynamic stage UI** — Stages should be driven from data, support arbitrary count. Currently some pages hardcode stage names.
- [ ] **Session state cleanup** — Too much stored in session state. Review and minimize.
- [ ] **Move inline plot code to plots module** — Business lever analysis (page 11) has plots defined inline.

### Restructure (align with IA app pattern)

- [ ] **Simplify Home.py** — Follow IA pattern: data source selector → load function → success/error. Current Home.py is cleaner now but still has legacy TODOs.
- [ ] **Simplify da_streamlit_utils.py** — Reduce complexity, possibly reuse `streamlit_utils.py` shared code.

---

## Streamlit Pages

### Page 1: Global Filters
- [ ] Customize filter dropdown (focus on logical filter fields)
- [ ] Add reset button for filters
- [ ] Make code robust against heavy filtering (e.g. dropping stages)
- [ ] Apply filters by default when set

### Page 2: Global Dashboard
- [ ] Add Value Finder-style pie chart at arbitration
- [ ] Speed up first-use (does too much computation upfront)

### Page 3: Action Distribution
- [ ] Dynamic stages from data
- [ ] Top-K limiter for bar charts
- [ ] Show all ticks or indicate truncation
- [ ] Consider multi-stage selection

### Page 4: Action Funnel
- [ ] Toggle between Passing and Filtered perspective
- [ ] Dynamic stage count
- [ ] Better coloring for filter components
- [ ] Move top-N control from sidebar to section
- [ ] Handle many stages gracefully

### Page 5: Global Sensitivity
- [ ] Infer top-X from data (max rank per channel for Final records)
- [ ] Win_rank upper bound from data, not hardcoded 10

### Page 6: Win/Loss Analysis
- [ ] Verify numbers (bar charts vs box plots)
- [ ] Consistent colors across paired charts
- [ ] Handle many channels
- [ ] Generalize rank usage
- [ ] Fix pale colors for small counts

### Page 7: Personalization Analysis
- [ ] Overlay propensity + priority on optionality plot
- [ ] Consistent stage color scheme
- [ ] Fix session state key warning

### Pages 8-9 (Unfinished)
- [ ] **Page 8: Offer Quality** — needs generalization, session state cleanup, move logic to class
- [ ] **Page 9: Thresholding** — largely broken. Fix: explode issue with num_samples > 1, filtering, interactive thresholds. Consider rewrite or removal.

### Page 10: Arbitration Component Distribution
- [ ] Finish proposition distribution side-by-side view

### Page 11: Business Lever Analysis
- [ ] Clean up session state usage
- [ ] Use aggregated/sampled data (not raw `decision_data`)
- [ ] Refactor lever calculation into DecisionAnalyzer class
- [ ] Move plots to plots module
- [ ] Start target win ratio at > 0

---

## Resolved / No Longer Applicable

- ~~Anonymization checkbox~~ — Users upload their own data now
- ~~Audience selection filter~~ — No audience concept in current data
- ~~`get_git_version_and_date()` dead code~~ — Removed
- ~~Hardcoded `app_version`/`tag_date`~~ — Now uses `pdstools.__version__`
- ~~`with_row_count` deprecation~~ — Fixed
- ~~`pl.count()` deprecation~~ — Fixed
- ~~`.melt()` deprecation~~ — Fixed
- ~~`Categorical(ordering=)` deprecation~~ — Fixed
- ~~`ColumnNotFoundError` import~~ — Fixed
- ~~`print()` in sample()~~ — Replaced with logger
- ~~`.columns` performance warning~~ — Fixed

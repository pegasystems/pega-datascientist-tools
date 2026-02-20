# Decision Analyzer — Refactoring Specs & Work Items

Tracked on branch: `refactor/decision-analyzer`

## Completed

- [x] **Test harness**: 98 unit tests covering v1/v2 core functionality (including component impact/drilldown tests)
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

- [ ] **Pre-aggregated data visualization** — Some exports contain pre-computed analysis results (sensitivity scores, action funnel data, decision summaries) rather than raw interaction-level data. Currently unsupported. Could bypass the DecisionAnalyzer and directly visualize the pre-aggregated parquet/csv files in the appropriate pages.
- [ ] **Graceful degradation for minimal data** — Some exports only contain context keys (Issue, Group, Name, InteractionID, SubjectID) without scoring columns (Priority, Propensity, Value, pxDecisionTime). The app currently crashes with a ColumnNotFoundError. Should show a clear message about which columns are missing and ideally support a reduced analysis mode (e.g. action distribution only, no sensitivity/optionality).
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

- [x] **Simplify Home.py** — Now follows the shared pattern: `standard_page_config()` → `show_version_header()` → data source selector → load function → success/error. Removed `os` import, dropped `os.getcwd() == "/app"` check.
- [x] **Simplify da_streamlit_utils.py** — Fixed backwards dependency: moved `get_current_index` to shared `streamlit_utils.py`, re-exported from `da_streamlit_utils` for backward compat. Removed dead commented-out wrappers, cleaned imports. Accepted parquet uploads. Still has complex `get_data_filters()` (200 lines) — to be split later.

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
- [x] **Deeper filter component / strategy analysis (partial)** — Added `getComponentActionImpact()` (component × action matrix), `getComponentDrilldown()` (per-component detail with scoring context from surviving rows), enhanced `getFilterComponentData()` with `pxComponentType`. Added corresponding plot methods (`component_action_impact`, `component_drilldown`, enhanced `filtering_components` with component type coloring). Integrated into Streamlit page 4 (Action Funnel). Remaining: component overlap analysis, component impact over time (trend).

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

### Page 8: Offer Quality (⚠ incomplete)
- [ ] Needs generalization, session state cleanup, move logic to class
- [ ] Pages moved from `unfinished_pages/` subfolder into regular `pages/` — unfinishedness tracked here

### Page 9: Thresholding (⚠ incomplete)
- [ ] Largely broken. Fix: explode issue with num_samples > 1, filtering, interactive thresholds
- [ ] Consider rewrite or removal
- [ ] Pages moved from `unfinished_pages/` subfolder into regular `pages/` — unfinishedness tracked here

### Page 10: Arbitration Component Distribution
- [ ] Finish proposition distribution side-by-side view

### Page 11: Business Lever Analysis
- [ ] Clean up session state usage
- [ ] Use aggregated/sampled data (not raw `decision_data`)
- [ ] Refactor lever calculation into DecisionAnalyzer class
- [ ] Move plots to plots module
- [ ] Start target win ratio at > 0

---

---

## Sampling & Large Data

Current approach: hash-based interaction sampling (`pxInteractionID.hash() % 1000 < threshold`). Keeps all actions within a sampled interaction together. Default 50,000 interactions. Used for expensive analyses (sensitivity, reranking, optionality); pre-aggregation runs on full data.

- [ ] **Fix docstring** — `sample()` says "taking the first 50,000 interactions" but actually uses hash-based sampling. Update to match implementation.
- [ ] **Stratified sampling option** — Current sampling is purely random by interaction ID. Consider optional stratification by channel, direction, or other categorical dimensions to ensure proportional representation in the sample. Important when channel distribution is skewed.
- [ ] **Early-stage sampling for large files** — When input exceeds a threshold (e.g. estimated >5M rows from file size or initial scan), auto-downsample during the file reading phase in the app before the DecisionAnalyzer constructor runs. Show a warning explaining the downsampling.
- [ ] **Streaming pre-aggregation** — `getPreaggregatedFilterView` currently `.collect()`s the full dataset. For GB-scale data this is the memory bottleneck. Investigate polars streaming engine or chunked processing to avoid loading everything into memory.
- [ ] **Data size warning in UI** — Show a warning when uploaded data exceeds a practical size threshold (e.g. >500 MB, >5M rows), with guidance on expected load times and memory usage.

---

## Technical Debt

_(No open items — see Resolved section below.)_

---

## Cross-App Consistency

Shared infrastructure added to `streamlit_utils.py` and adopted by all three apps (Health Check, Impact Analyzer, Decision Analyzer).

- [x] **`standard_page_config()`** — Consistent `set_page_config` with `layout="wide"`, shared `menu_items` (bug report + docs links). All Home pages and sub-pages use it.
- [x] **`show_version_header()`** — Displays `pdstools {version}` caption with upgrade hint. Checks PyPI for latest version and shows a warning if outdated. All Home pages call it.
- [x] **`ensure_session_data()`** — Shared guard function. DA uses via `ensure_data()`, IA via `ensure_impact_analyzer()`.
- [x] **`--deploy-env` CLI flag** — `cli.py` accepts `--deploy-env ec2` (or any value), propagates as `PDSTOOLS_DEPLOY_ENV` env var. DA reads via `get_deploy_env()` / `is_managed_deployment()`. Replaces `os.getcwd() == "/app"` hack. EC2 sample path configurable via `PDSTOOLS_SAMPLE_DATA_PATH` env var.
- [x] **Unified data source labels** — All apps use "Sample data", "File upload", "File path".
- [x] **DA file upload expanded** — Now accepts `zip, parquet, json, csv, arrow` (was `zip, parquet` only).
- [x] **IA sys.path hack removed** — Home.py and all pages no longer manipulate `sys.path`.
- [x] **Consistent welcome text** — All Home pages have a clean title, brief description, and version header with upgrade hint.
- [ ] **HC data import alignment** — Health Check still uses its own `import_datamart()` pattern in a separate page with different labels ("Direct file path", "CDH Sample", etc.). Could be aligned further in a follow-up.

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
- ~~Hardwired EC2 paths~~ — Replaced `os.getcwd() == "/app"` with `--deploy-env` CLI flag and `PDSTOOLS_DEPLOY_ENV` env var. Sample path configurable via `PDSTOOLS_SAMPLE_DATA_PATH`.

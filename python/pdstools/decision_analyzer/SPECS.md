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

- [x] **Human-friendly field names (display_name refactor)** — Internal Pega field names (`pyIssue`, `pxInteractionID`, etc.) replaced with friendly names ("Issue", "Interaction ID", "Channel", etc.) on all UI surfaces. `table_definition.py` is the single source of truth via `display_name`. See dedicated section below for details. Remaining internal names (`is_mandatory`, `day`) are internal working columns (expected exceptions). `pxRank` has been renamed to `Rank`.
- [x] **Handle missing PVCL components** — `reRank()` now checks which columns exist and fills missing ones with 1.0 (neutral for multiplication).
- [ ] **Fix num_samples > 1** — Pre-aggregation sampling locked at 1 because >1 breaks `.explode()` in thresholding. Either fix thresholding or document limitation.
- [ ] **Validate fields_for_data_filtering** — Subset against actual available columns instead of hardcoded list. Update to use display names after the friendly-names refactor.
- [ ] **Human-friendly stage names** — Map internal names to display names (e.g. "Final" → "Presented"). May need a config dict.
- [x] **Channels + direction** — `get_overview_stats` now counts unique channel/direction combinations (falls back to channel-only if pyDirection absent).

### Medium Priority

- [x] **Move `filtered_action_counts` into class** — Standalone function in `utils.py` that duplicates the `aggregate_remaining_per_stage` pattern. Now a method on `DecisionAnalyzer`.
- [ ] **Refactor `get_offer_quality`** — Uses a manual stage loop; should delegate to `aggregate_remaining_per_stage`.
- [ ] **Win_rank flexibility** — `get_win_loss_distribution_data` has fixed rank parameter. Return all ranks, let UI filter.
- [ ] **Caching for parameterized methods** — `getThresholdingData` and `get_sensitivity` are expensive but not cached. Consider `lru_cache` or polars caching.
- [x] **DRY prio factor list** — `PRIO_FACTORS` constant defined in `utils.py`, used by `DecisionAnalyzer.py` and `plots.py`.

### Low Priority

- [ ] **Pre-aggregated data visualization** — Some exports contain pre-computed analysis results (sensitivity scores, action funnel data, decision summaries) rather than raw interaction-level data. Currently unsupported. Could bypass the DecisionAnalyzer and directly visualize the pre-aggregated parquet/csv files in the appropriate pages.
- [ ] **Graceful degradation for minimal data** — Some exports only contain context keys (Issue, Group, Name, InteractionID, SubjectID) without scoring columns (Priority, Propensity, Value, pxDecisionTime). The app currently crashes with a ColumnNotFoundError. Should show a clear message about which columns are missing and ideally support a reduced analysis mode (e.g. action distribution only, no sensitivity/optionality).
- [ ] **Add treatment to scope hierarchy** — Currently commented out in `NBADScope_Mapping`. Will be addressed as part of the friendly-names refactor (treatment becomes a display name in the data dictionary).
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

### Page 1: Global Data Filters
- [x] **Rename to Global Data Filters** — Page file renamed from `1_Global_Filters.py` to `1_Global_Data_Filters.py`, title updated.
- [x] **Add Interactions to filter summary** — `get_first_level_stats()` now reports Actions (unique Issue/Group/Action combos), Interactions (unique Interaction IDs, i.e. decisions), and Rows (total dataset rows). Removed stale "sampling settings" remark from page text.
- [x] **Fix deprecated polars serialization API** — Replaced `Expr.from_json` → `Expr.deserialize` and `meta.write_json` → `meta.serialize` (aligned with Health Check page).
- [ ] Customize filter dropdown (focus on logical filter fields)
- [ ] Add reset button for filters
- [ ] Make code robust against heavy filtering (e.g. dropping stages)
- [ ] Apply filters by default when set

### Page 2: Overview
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

## Human-Friendly Field Names (`display_name` refactor)

**Goal**: Users never see internal Pega field names (`pyIssue`, `pxInteractionID`, etc.) in the UI. Every column uses a user-friendly name like "Issue", "Interaction ID", "Channel". The data dictionary (`table_definition.py`) is the single source of truth — changing a name there propagates everywhere.

### Current state (at time of writing — now completed)

- `TableConfig` used a `label` field. Most labels were just the internal Pega name again (`"pyIssue"` → label `"pyIssue"`), not user-friendly.
- `NBADScope_Mapping` in `utils.py` was a parallel dict — a second source of truth for display names.
- The filter multiselect in `get_data_filters()` had a `format_func` hack that strips `py`/`px` prefixes — a third approach.
- Internal names were hardcoded in polars expressions across `DecisionAnalyzer.py`, `utils.py`, `plots.py`, all Streamlit pages, and tests.

### Design

Replace `label` with `display_name`. Single-pass rename at ingestion. All downstream code uses friendly names.

**`TableConfig` before → after:**
```python
# BEFORE
"pyIssue": {"label": "pyIssue", "default": True, "type": pl.Categorical, "aliases": ["Issue"]}

# AFTER
"pyIssue": {"display_name": "Issue", "default": True, "type": pl.Categorical, "aliases": ["Issue"]}
```

The raw column key (`"pyIssue"`) identifies what arrives in the data. The `display_name` (`"Issue"`) is what the column is renamed to. `aliases` lists additional incoming names that should also map to this entry (e.g. data that already uses `"Issue"` instead of `"pyIssue"`).

**Full rename mapping (representative entries):**
| Raw key | display_name |
|---|---|
| `pyIssue` | Issue |
| `pyGroup` | Group |
| `pyName` | Action |
| `pxInteractionID` | Interaction ID |
| `pxDecisionTime` | Decision Time |
| `Primary_pySubjectID` | Subject ID |
| `Primary_ContainerPayload_Channel` | Channel |
| `Primary_ContainerPayload_Direction` | Direction |
| `pxComponentName` | Component Name |
| `pxComponentType` | Component Type |
| `Stage_pyName` | Stage |
| `Stage_pyStageGroup` | Stage Group |
| `Stage_pyOrder` | Stage Order |
| `pxStrategyName` | Strategy Name |
| `pyTreatment` | Treatment |

### Implementation steps

- [x] **Replace `label` with `display_name`** in `TableConfig` TypedDict and all entries in both `DecisionAnalyzer` and `ExplainabilityExtract` dicts in `table_definition.py`.
- [x] **Simplify `ColumnResolver`** — single rename pass: raw column key → `display_name`. Remove the intermediate `label` concept. When both raw key and display_name exist as columns, prefer display_name (existing conflict resolution logic).
- [x] **Update `resolve_aliases`** — target is now `display_name` instead of the raw key.
- [x] **Update all `pl.col()` references** — mechanical search-and-replace across `DecisionAnalyzer.py`, `utils.py`, `plots.py`, all Streamlit pages, `da_streamlit_utils.py`, and tests. Remaining `is_mandatory` / `day` references are internal working columns (expected). `pxRank` has been renamed to `Rank`.
- [x] **Retire `NBADScope_Mapping`** — removed from `utils.py` and all plot code.
- [x] **Remove `format_func` hack** in `get_data_filters()` — multiselect shows column names directly.
- [x] **Update `fields_for_data_filtering`** — uses display names.
- [x] **Update `preaggregation_columns`** — uses display names.
- [x] **Update `audit_tag_mapping`** — no longer exists; not applicable.
- [x] **Update tests** — assertions and column references in `test_DecisionAnalyzer.py` updated.

### Risks & notes

- This is a large, invasive change but purely mechanical. Best done in a single PR to avoid half-renamed state.
- The `aliases` field ensures backward compatibility: data arriving with either internal names or friendly names will work.
- Stage names (`"Arbitration"`, `"Output"`, `"AvailableActions"`) are **not** renamed in this pass — those are already readable. Stage-name customization (e.g. "Final" → "Presented") is tracked as a separate item.

---

## CLI Pre-Ingestion Sampling (`--sample`)

**Goal**: For large datasets, sample data **before** it enters the `DecisionAnalyzer` constructor, reducing memory usage and startup time. Sampling is always stratified on Interaction ID so all decisions within an interaction stay together.

### Current state

- `DecisionAnalyzer` has a `sample` cached property that creates a **secondary** subset (default 50k interactions) for expensive analyses (sensitivity, reranking, optionality). The full data is always loaded into `decision_data` and used for pre-aggregation.
- No way to limit data size from the CLI or before the constructor runs.
- The existing hash-based sampling logic in `sample` is proven and fast.

### Design

Add a `--sample` CLI flag and a shared `sample_interactions()` utility. The CLI passes the limit via env var. Home.py applies the sampling to the raw LazyFrame before constructing DecisionAnalyzer.

**CLI usage:**
```bash
# Absolute count — keep at most 100k interactions (all their rows)
pdstools decision_analyzer --sample 100000

# Percentage — keep ~10% of interactions
pdstools decision_analyzer --sample 10%
```

**Utility function** (`decision_analyzer/utils.py`):
```python
def sample_interactions(
    df: pl.LazyFrame,
    n: Optional[int] = None,
    fraction: Optional[float] = None,
    id_column: str = "Interaction ID",
) -> pl.LazyFrame:
```
- Exactly one of `n` or `fraction` must be provided.
- Uses hash-based filtering: `col(id_column).hash() % 10000 < threshold`. No `.collect()` needed — the filter pushes down into the scan.
- Deterministic: same data + same limit = same sample.

**Data flow:**
```
CLI --sample 100000
  → env var PDSTOOLS_SAMPLE_LIMIT=100000
    → Home.py reads env var
      → raw_data = sample_interactions(raw_data, n=100000)
        → DecisionAnalyzer(raw_data, ...)
```

### Implementation steps

- [ ] **Add `sample_interactions()` utility** in `decision_analyzer/utils.py`. Reuse the hash-based approach from the existing `sample` property.
- [ ] **Add `--sample` CLI flag** to `create_parser()` in `cli.py`. Parse: if value ends in `%`, treat as fraction; otherwise as absolute count. Propagate as `PDSTOOLS_SAMPLE_LIMIT` env var.
- [ ] **Apply in Home.py** — read `PDSTOOLS_SAMPLE_LIMIT` env var. If set, apply `sample_interactions()` to `raw_data` before passing to `load_decision_analyzer()`. Show an `st.info()` banner explaining the pre-sampling.
- [ ] **Update the internal `sample` property docstring** — clarify that it operates on potentially pre-sampled data, so two layers of reduction are possible.
- [ ] **Note on column naming**: the `id_column` parameter defaults to the post-rename display name. If sampling is applied before `rename_and_cast_types`, pass the raw column name. The friendly-names refactor determines the exact default — either `"Interaction ID"` (post-rename) or `"pxInteractionID"` (pre-rename). Since sampling happens on the raw LazyFrame in Home.py before DA construction, and alias resolution hasn't run yet, the utility should try common names (`pxInteractionID`, `InteractionID`, `Interaction ID`) and use whichever exists.

### Existing sampling items (updated)

- [ ] **Fix docstring** — `sample()` says "taking the first 50,000 interactions" but actually uses hash-based sampling. Update to match.
- [ ] **Stratified sampling option** — Current `sample` property is purely random by interaction ID. Consider optional stratification by channel/direction for proportional representation. Separate from the CLI pre-ingestion sampling.
- [ ] **Streaming pre-aggregation** — `getPreaggregatedFilterView` currently `.collect()`s the full dataset. For GB-scale data this is the memory bottleneck. Investigate polars streaming engine or chunked processing.
- [ ] **Data size warning in UI** — Show a warning when uploaded data exceeds a practical size threshold (e.g. >500 MB, >5M rows).

---

## CLI Data Path (`--data-path`)

**Goal**: Provide a CLI option to specify the path to decision data. Useful for managed installs (Docker, EC2) where data is at a known location. The path shows in Home.py as an alternative data source.

### Current state

- `PDSTOOLS_SAMPLE_DATA_PATH` env var points to a hardcoded S3-mounted path (`/s3-files/anonymized/anonymized`). Only used when `is_managed_deployment()` is true.
- `handle_file_path()` in `da_streamlit_utils.py` shows a text input for manual path entry, but only in managed deployments.
- `_EC2_SAMPLE_PATH` constant in `da_streamlit_utils.py` is the default.

### Design

Replace `PDSTOOLS_SAMPLE_DATA_PATH` and `_EC2_SAMPLE_PATH` with a single `--data-path` CLI flag. The path is shown in Home.py as an alternative data source alongside file upload.

**CLI usage:**
```bash
# Docker/EC2 deployment
pdstools decision_analyzer --deploy-env ec2 --data-path /s3-files/data

# Local use — point at a directory
pdstools decision_analyzer --data-path /Users/me/exports/latest/
```

**Data loading priority in Home.py:**
1. **File upload** — always visible, user drags/drops files.
2. **Configured data path** — if `--data-path` was given, show a section: `📂 Configured data path: /path/to/data` with a load button. In managed deployments, auto-load if no file was uploaded.
3. **Sample data** — fallback to GitHub-hosted demo data (only when neither upload nor data path produced data).

### Implementation steps

- [ ] **Add `--data-path` CLI flag** to `create_parser()` in `cli.py`. Propagate as `PDSTOOLS_DATA_PATH` env var.
- [ ] **Retire `PDSTOOLS_SAMPLE_DATA_PATH`** env var and `_EC2_SAMPLE_PATH` constant in `da_streamlit_utils.py`.
- [ ] **Add `get_data_path()` helper** in `streamlit_utils.py` — reads `PDSTOOLS_DATA_PATH` env var, returns `Optional[str]`.
- [ ] **Update Home.py data loading flow** — after file upload, check `get_data_path()`. If set and no upload, show the configured path section and load from it. In managed deployments, auto-load without requiring a button click.
- [ ] **Update `handle_sample_data()`** — remove the `_EC2_SAMPLE_PATH` branch. Managed deployments use `--data-path` instead.
- [ ] **Update Cross-App Consistency section** — mark `PDSTOOLS_SAMPLE_DATA_PATH` as retired, document `--data-path`.

---

## Technical Debt

_(No open items — see Resolved section below.)_

---

## Cross-App Consistency

Shared infrastructure added to `streamlit_utils.py` and adopted by all three apps (Health Check, Impact Analyzer, Decision Analyzer).

- [x] **`standard_page_config()`** — Consistent `set_page_config` with `layout="wide"`, shared `menu_items` (bug report + docs links). All Home pages and sub-pages use it.
- [x] **`show_version_header()`** — Displays `pdstools {version}` caption with upgrade hint. Checks PyPI for latest version and shows a warning if outdated. All Home pages call it.
- [x] **`ensure_session_data()`** — Shared guard function. DA uses via `ensure_data()`, IA via `ensure_impact_analyzer()`.
- [x] **`--deploy-env` CLI flag** — `cli.py` accepts `--deploy-env ec2` (or any value), propagates as `PDSTOOLS_DEPLOY_ENV` env var. DA reads via `get_deploy_env()` / `is_managed_deployment()`. Replaces `os.getcwd() == "/app"` hack.
- [ ] **`--data-path` CLI flag** — Replace `PDSTOOLS_SAMPLE_DATA_PATH` env var and `_EC2_SAMPLE_PATH` constant with a proper `--data-path` CLI flag. See dedicated section above.
- [ ] **`--sample` CLI flag** — Pre-ingestion interaction sampling for large datasets. See dedicated section above.
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

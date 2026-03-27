# Decision Analyzer Architecture Summary

**Date:** March 19, 2026
**Purpose:** Understanding the current Decision Analyzer architecture for analysis and potential feature additions

---

## 1. CLI Structure for Decision Analyzer

### Entry Point
**File:** [python/pdstools/cli.py](python/pdstools/cli.py)

The CLI is a unified entry point for all pdstools apps, including Decision Analyzer (alias: `da`).

#### CLI Flags Available

| Flag | Type | Purpose | Exposed As |
|------|------|---------|-----------|
| `--data-path PATH` | str, optional | Path to data file/directory to load on startup. Supports parquet, csv, json, arrow, zip, tar, tar.gz, tgz, and partitioned folders | `PDSTOOLS_DATA_PATH` env var |
| `--sample SPEC` | str, optional | Pre-ingestion interaction sampling. Accepts absolute count (`100k`, `1M`) or percentage (`10%`). All rows for each sampled interaction are kept. | `PDSTOOLS_SAMPLE_LIMIT` env var |
| `--temp-dir DIR` | str, optional | Directory for temporary files (sampled data parquet, cache). Defaults to current working directory. | `PDSTOOLS_TEMP_DIR` env var |

**Execution Examples:**
```bash
# Launch Decision Analyzer with data pre-loaded and sampled
pdstools decision_analyzer --data-path /path/to/data --sample 100k

# Or using alias
pdstools da --data-path /path/to/data --sample 10%
```

**Command flow:**
1. CLI parses flags → creates `PDSTOOLS_*` environment variables
2. Streamlit app reads env vars via [python/pdstools/utils/streamlit_utils.py](python/pdstools/utils/streamlit_utils.py) helpers
3. Home.py initializes data loading based on env vars

---

## 2. Data Loading Architecture

### Multi-Format Core Loader
**File:** [python/pdstools/pega_io/File.py](python/pdstools/pega_io/File.py) (referenced from decision_analyzer)

Core function: `read_data(path)` handles multiple formats transparently:
- **Parquet** (single file or Hive-partitioned directory)
- **CSV, JSON, NDJSON**
- **Arrow files**
- **ZIP archives** (auto-extracts to temp directory)
- **TAR/TAR.GZ/TGZ** (auto-extracts to temp directory)
- **Pega Action Analysis exports** (nested ZIP with gzipped NDJSON)

Returns: `pl.LazyFrame` for lazy evaluation

### Decision Analyzer Specific Utilities
**File:** [python/pdstools/decision_analyzer/data_read_utils.py](python/pdstools/decision_analyzer/data_read_utils.py)

Specialized functions:
- `read_nested_zip_files(buffer)` — Handles Pega Action Analysis export format (ZIP containing inner `.zip` files that are actually gzipped NDJSON)
- `read_gzipped_data(buffer)` — Decompresses and reads individual gzipped NDJSON chunks
- `read_gzipped_ndjson_directory(path)` — Reads directories of gzipped NDJSON files
- `validate_columns(data, table_def)` — Validates required columns are present

### Streamlit Integration Layer
**File:** [python/pdstools/app/decision_analyzer/da_streamlit_utils.py](python/pdstools/app/decision_analyzer/da_streamlit_utils.py) (lines 1-100+)

Key functions:
- **`handle_file_upload()`** — Multi-file uploader supporting all formats and archives with progress feedback
- **`handle_data_path()`** — Loads from `--data-path` CLI flag with spinner feedback
- **`handle_sample_data()`** — Loads built-in EEV2 demo data (sample dataset)
- **`load_decision_analyzer(raw_data)`** — Wrapper to create DecisionAnalyzer instance from loaded data

### Data Flow from CLI to DecisionAnalyzer

```
┌──────────────────────────────────────────────────────────────┐
│ CLI (cli.py)                                                  │
│ • Parses --data-path, --sample, --temp-dir flags             │
│ • Creates environment variables: PDSTOOLS_*                  │
└──────────────────────────────────┬──────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Home.py (Streamlit UI)                                        │
│ • Reads env vars via get_data_path(), get_sample_limit()     │
│ • User uploads file (optional)                               │
│ • Calls handle_file_upload() OR handle_data_path()           │
│                                                               │
│ Optional: Pre-ingestion sampling via prepare_and_save()      │
│  - Uses parse_sample_flag() to parse --sample spec           │
│  - Applies sample_interactions() with hash-based sampling    │
│  - Saves sampled parquet to temp directory                   │
└──────────────────────────────────┬──────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────┐
│ DecisionAnalyzer class (__init__)                             │
│ • Receives raw_data: pl.LazyFrame                            │
│ • Column renaming & validation (column_schema.py)            │
│ • Data type casting and cleanup                              │
│ • Creates unfiltered_raw_decision_data                       │
│ • Initializes decision_data = unfiltered_raw_decision_data   │
└──────────────────────────────────┬──────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Session State (Streamlit)                                     │
│ • st.session_state.decision_data = DecisionAnalyzer instance │
│ • st.session_state.sample_metadata = sampling info           │
│ • Shared across all analysis pages                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. DecisionAnalyzer Class

### Location & Import
**File:** [python/pdstools/decision_analyzer/DecisionAnalyzer.py](python/pdstools/decision_analyzer/DecisionAnalyzer.py)
**Import:** `from pdstools.decision_analyzer import DecisionAnalyzer`

### Class Methods for Initialization

#### Factory Methods
```python
DecisionAnalyzer.from_explainability_extract(source: str | os.PathLike) → DecisionAnalyzer
DecisionAnalyzer.from_decision_analyzer(source: str | os.PathLike) → DecisionAnalyzer
```

#### Direct Constructor
```python
DecisionAnalyzer(
    raw_data: pl.LazyFrame,
    level: str = "Stage Group",
    sample_size: int = 10_000,
    mandatory_expr: pl.Expr | None = None,
    additional_columns: dict[str, pl.DataType] | None = None
)
```

**Parameters:**
| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `raw_data` | `pl.LazyFrame` | *required* | Raw decision data from Explainability Extract or EEV2 |
| `level` | str | `"Stage Group"` | Granularity: `"Stage Group"` (synthetic groups) or `"Stage"` (individual stages, v2 only) |
| `sample_size` | int | `10,000` | Max unique interactions to sample for distribution analyses. Higher = more accurate but slower. |
| `mandatory_expr` | `pl.Expr` | None | Polars expression to mark mandatory actions (e.g., `pl.col("Issue") == "Retention"`) for ranking |
| `additional_columns` | dict | None | Extra columns beyond standard schema (e.g., custom fields) |

**Validation on Init:**
- Critical columns checked: `"Interaction ID"`, `"Issue"`, `"Group"`, `"Action"` **must be present** (raises ValueError if missing)
- Column aliasing resolves both Pega internal names and display names
- Data type casting applied per column_schema.py definitions
- Format auto-detected: `extract_type = "explainability_extract"` vs `"decision_analyzer"`

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `decision_data` | `pl.LazyFrame` | Current working dataset (with global filters applied if any) |
| `unfiltered_raw_decision_data` | `pl.LazyFrame` | Original data before any global filters, always available for reset |
| `extract_type` | str | Either `"explainability_extract"` (v1, arbitration-only) or `"decision_analyzer"` (v2, full pipeline) |
| `level` | str | Current stage granularity level (`"Stage Group"` or `"Stage"`) |
| `sample_size` | int | Maximum interactions to sample |
| `sample` | `pl.LazyFrame` (@cached_property) | Hash-based deterministic sample of unique interactions. All rows within selected interactions kept. |
| `AvailableNBADStages` | list[str] | All stage values in the data at current `level` (e.g., `["Filtering", "Arbitration", "Output"]`) |
| `fields_for_data_filtering` | list[str] | Columns available for global filtering (subset of default list) |
| `color_mappings` | dict[str, dict[str, str]] (@cached_property) | Pre-computed consistent color assignments for all categorical dimensions |
| `plot` | Plot | Plot accessor for visualization methods (see plots.py) |

### Major Cached Properties & Methods

#### Key Cached Properties (automatically invalidated on filter/level changes)
```python
@cached_property
def sample() → pl.LazyFrame
    # Hash-based sample of unique interactions

@cached_property
def color_mappings() → dict[str, dict[str, str]]
    # Consistent colors for Issue, Group, Action, Channel, Direction, Stage, etc.

@cached_property
def stages_with_propensity() → list[str]
    # Stages with meaningful (non-default) propensity values

@cached_property
def stages_from_arbitration_down() → list[str]
    # All stages from Arbitration onward

@cached_property
def arbitration_stage() → pl.LazyFrame
    # Sample filtered to arbitration stage and later

@cached_property
def getPreaggregatedFilterView() → pl.LazyFrame
    # Pre-aggregated view showing what gets filtered at each stage
    # Grouped by Issue, Group, Action, Treatment, Channel, Direction, etc.
    # Includes Win_at_rank1..5 columns for ranking analysis

@cached_property
def getPreaggregatedRemainingView() → pl.LazyFrame
    # Pre-aggregated view showing remaining offers per stage
```

#### Key Query Methods
```python
# Filtering & configuration
applyGlobalDataFilters(filters: pl.Expr | list[pl.Expr]) → None
resetGlobalDataFilters() → None
set_level(level: str) → None

# Data retrieval
getDistributionData(stage, grouping_levels, additional_filters) → pl.LazyFrame
getFunnelData(scope, additional_filters) → tuple[pl.LazyFrame, pl.DataFrame]
getFilterComponentData(top_n, additional_filters) → pl.DataFrame
getComponentDrilldown(component_name, additional_filters) → pl.DataFrame
getComponentActionImpact(top_n, scope, additional_filters) → pl.DataFrame

# Analysis data
overview_stats → dict[str, ...]  (also available as get_overview_stats for backward compat)
get_win_loss_distribution_data(level, win_rank, additional_filters) → pl.LazyFrame
get_optionality_data(df) → dict[str, ...]
get_optionality_funnel(df) → ...
getThresholdingData(field, quantile_range) → dict[tuple, pl.DataFrame]
get_sensitivity(win_rank, group_filter, additional_filters) → pl.LazyFrame
get_winning_or_losing_interactions(win_rank, group_filter, win, additional_filters) → pl.DataFrame
get_trend_data(level, groupby, additional_filters) → pl.LazyFrame

# Utilities
getPossibleScopeValues() → list[str]  # ["Issue", "Group", "Action"]
getPossibleStageValues() → list[str]  # Stages in current level
getAvailableFieldsForFiltering(categoricalOnly=False) → list[str]
```

#### Ranking Function
Actions are ranked within each interaction using this priority:
1. `is_mandatory` (1 = first, 0 = later) — set via `mandatory_expr` parameter
2. `Priority` (descending) — numeric propensity or score
3. `Stage Order` (descending) — position in pipeline
4. `Issue` rank, `Group` rank, `Action` rank (alphabetic tiebreaker)

**Output:** `Rank` column in data (1 = highest priority action in that interaction)

---

## 4. Current Analyses (Streamlit Pages)

### Home Page
**File:** [python/pdstools/app/decision_analyzer/Home.py](python/pdstools/app/decision_analyzer/Home.py)

**Purpose:** Data loading and configuration
**Key Features:**
- File upload widget (supports all formats + archives)
- Loads from `--data-path` CLI flag if provided
- Falls back to built-in sample data if nothing uploaded
- Pre-ingestion sampling via `--sample` flag
- Sample size slider for distribution overview calculations (separate from pre-ingestion sampling)
- Displays loading status and sampling progress with time estimates

---

### 2. Global Data Filters
**File:** [python/pdstools/app/decision_analyzer/pages/1_Global_Data_Filters.py](python/pdstools/app/decision_analyzer/pages/1_Global_Data_Filters.py)

**Purpose:** Apply cross-cutting filters to all downstream analyses
**Key Features:**
- Multi-select widgets for available filter fields (Channel, Issue, Group, Action, Direction, etc.)
- Filter composition (AND logic)
- Save/load filters as JSON files
- Persistent filter state across page navigation
- All analysis pages respect these global filters
- Colors remain consistent despite filtering

**Data Need:** Full dataset (via `unfiltered_raw_decision_data`)

---

### 3. Overview
**File:** [python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py](python/pdstools/app/decision_analyzer/pages/3_Overview.py) → Actually at [3_Overview.py](python/pdstools/app/decision_analyzer/pages/3_Overview.py)

**Purpose:** High-level summary metrics and insights
**Key Features:**
- **Source Data:** Total actions, channels, duration, decision count, unique customers
- **Customer Choice:** Propensity vs. optionality plot (actions offered vs. likelihood to respond)
- **What Drives Offers:** Component importance / lever analysis
- **Offer Quality:** Win/loss distribution pie chart
- Detects best stage for propensity analysis automatically
- Shows sample information if pre-ingestion sampling applied
- Forces stage level to "Stage Group" for consistency

**Data Need:** Sample data at stage with propensity values, metadata

---

### 4. Action Distribution
**File:** [python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py](python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py)

**Purpose:** Analyze distribution of actions across dimensions
**Key Features:**
- Scope selector (Issue, Group, Action levels)
- Stage selector
- Color by dimension selector (Channel, Direction, etc.)
- Histograms and bar charts
- Propensity and priority distributions
- Component analysis

**Data Need:** Sample data filtered by stage, aggregated by scope

---

### 5. Action Funnel
**File:** [python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py](python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py)

**Purpose:** Track how offers flow through decision pipeline
**Key Features:**
- **Remaining Tab:** Actions making it through each stage (passing all filters)
- **Filtered Tab:** Actions eliminated at each stage
- Scope selector (Issue/Group/Action)
- Channel/direction filters
- Shows:
  - Average actions per interaction (funnel height)
  - Reach % (% of interactions with ≥1 action at that scope)
- Component filtering view

**Data Need:** Pre-aggregated filter and remaining views, full dataset for component analysis

---

### 6. Global Sensitivity
**File:** [python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py](python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py)

**Purpose:** What-if analysis — how sensitive is ranking to priority/propensity adjustments
**Key Features:**
- Select dimension to vary (Priority, Propensity, Value, Context Weight, Levers)
- Adjust percentage increase/decrease
- Shows impact on winning actions
- Cached results for performance

**Data Need:** Sample data with ranking and propensity columns

---

### 7. Win Loss Analysis
**File:** [python/pdstools/app/decision_analyzer/pages/7_Win_Loss_Analysis.py](python/pdstools/app/decision_analyzer/pages/7_Win_Loss_Analysis.py)

**Purpose:** Analyze characteristics of winning vs. losing offers
**Key Features:**
- Win rank selector (1st, 2nd, 3rd, etc.)
- Drilldown by Issue, Group, Action
- Distribution charts for winning vs. losing actions
- Can compare metrics like priority, propensity, value across winners/losers

**Data Need:** Sample data with ranking, propensity, and dimension columns

---

### 8. Optionality Analysis
**File:** [python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py](python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py)

**Purpose:** How many options do customers get at each stage
**Key Features:**
- Optionality metrics (min, max, mean actions per interaction)
- Over-time trend analysis
- Distribution of action counts (how many customers see 1, 2, 3+ actions)
- Propensity correlation with optionality

**Data Need:** Sample data with Interaction ID grouping

---

### 9. Offer Quality Analysis
**File:** [python/pdstools/app/decision_analyzer/pages/9_Offer_Quality_Analysis.py](python/pdstools/app/decision_analyzer/pages/9_Offer_Quality_Analysis.py)

**Purpose:** Assess quality and diversity of offers
**Key Features:**
- Win distribution (rank 1 actions)
- Action variety at each rank
- Quality metrics by dimension (Channel, Group, etc.)
- Pie charts showing offer distribution

**Data Need:** Sample data with ranking columns

---

### 10. Thresholding Analysis
**File:** [python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py](python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py)

**Purpose:** How would different propensity/priority thresholds change offer selection
**Key Features:**
- Select field to threshold (Propensity, Priority, Value, etc.)
- Move slider across quantile range (10-90th percentile)
- Shows impact on action counts and reach
- Cached for performance

**Data Need:** Sample data with numeric columns

---

### 11. Arbitration Distribution
**File:** [python/pdstools/app/decision_analyzer/pages/11_Arbitration_Distribution.py](python/pdstools/app/decision_analyzer/pages/11_Arbitration_Distribution.py)

**Purpose:** Deep dive into arbitration stage specifically
**Key Features:**
- Not fully documented yet
- Likely focuses on Arbitration and later stages

---

### 12. About
**File:** [python/pdstools/app/decision_analyzer/pages/12_About.py](python/pdstools/app/decision_analyzer/pages/12_About.py)

**Purpose:** Help and version information
**Content:** Uses shared `show_about_page()` utility from streamlit_utils.py

---

## 5. Data Characteristics

### Column Schema
**File:** [python/pdstools/decision_analyzer/column_schema.py](python/pdstools/decision_analyzer/column_schema.py)

#### Critical Columns (Must Be Present)
| Column | Pega Internal Name | Display Name | Type | Notes |
|--------|-------------------|--------------|------|-------|
| Interaction ID | `pxInteractionID` | Interaction ID | Utf8 | **PRIMARY KEY** — Groups all actions within a customer interaction |
| Issue | `pyIssue` | Issue | Categorical | Business area (e.g., "Retention", "Sales") |
| Group | `pyGroup` | Group | Categorical | Sub-category of Issue |
| Action | `pyName` | Action | Utf8 | Individual offer/action name |

#### Key Analysis Columns
| Column | Pega Name | Type | Purpose |
|--------|-----------|------|---------|
| Decision Time | `pxDecisionTime` | Datetime | When decision was made, used for trends and sampling |
| Propensity | `FinalPropensity` | Float64 | Model's estimated likelihood of customer response (0-1) |
| Priority | `Priority` | Float32 | Ranking priority score |
| Value | `Value` | Float64 | Business value of action |
| Context Weight | `ContextWeight` | Float64 | Weight from contextual features |
| Levers | `Weight` | Float64 | Contribution from decision levers |
| Stage | `Stage_pyName` | Categorical | Individual strategy stage (v2 only) |
| Stage Group | `Stage_pyStageGroup` | Categorical | Synthetic grouping of stages |
| Stage Order | `Stage_pyOrder` | Int32 | Pipeline position for sorting |
| Component Name | `pxComponentName` | Utf8 | Filter component that eliminated action (if filtered) |
| Component Type | `pxComponentType` | Utf8 | Type of component (When rule, etc.) |
| Channel | `Primary_ContainerPayload_Channel` | Categorical | Communication channel (email, phone, etc.) |
| Direction | `Primary_ContainerPayload_Direction` | Categorical | Direction of interaction (inbound/outbound) |
| Treatment | `pyTreatment` | Utf8 | Treatment group (for A/B tests) |
| Record Type | (derived) | Categorical | `"OUTPUT"` (won) or `"FILTERED_OUT"` (lost) |
| Rank | (derived) | Int32 | Action ranking within interaction (1 = highest priority) |
| is_mandatory | (derived) | Int32 | Whether action marked mandatory (1/0) |

**Column Aliases:** Multiple names resolve to same display name (e.g., `pxInteractionID`, `InteractionID`, `Interaction ID` all map to "Interaction ID")

### Data Format Detection
The DecisionAnalyzer auto-detects format based on column presence:

| Detection | Format | Export Source | Scope |
|-----------|--------|----------------|-------|
| Has `Stage_pyName` + `Stage_pyStageGroup` | v2 / "decision_analyzer" | Action Analysis / EEV2 | Full decision pipeline with real stages |
| Missing stages columns | v1 / "explainability_extract" | Explainability Extract | Arbitration stage only; synthetic stages derived from ranking |

### Interaction Structure

```
One Interaction = One Customer Decision Point
├── Multiple Rows
│   ├── Action 1 (Issue=Retention, Group=Discount, Priority=0.85, Rank=1, Record Type=OUTPUT)
│   ├── Action 2 (Issue=Sales, Group=Upsell, Priority=0.72, Rank=2, Record Type=FILTERED_OUT)
│   ├── Action 3 (Rank=3, Record Type=FILTERED_OUT)
│   └── ...
├── All rows share same Interaction ID, Decision Time, Channel, Direction, Subject ID
└── Rank order determines which actions "won" (Rank=1) vs "lost"
```

### Data Scale Assumptions

| Analysis | Typical Size | Sampling Used | Notes |
|----------|--------------|-------------------|-------|
| Distribution overview (histograms, violin plots) | 10k interactions (default) | Yes, via `sample_size` param | Configurable in Home page |
| Global metrics (counts, averages) | Full dataset | No | Computed from full filtered data |
| Pre-aggregated views | ~100k aggregated rows | Depends on data | Much smaller than raw data |
| Component analysis | Full dataset | No | Expensive but necessary for complete view |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ CLI (python/pdstools/cli.py)                                     │
│ • --data-path, --sample, --temp-dir                             │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ Home.py (Django-style page routing)                             │
│ • File upload widget OR handle_data_path() OR sample data       │
│ • Pre-ingestion sampling (optional)                             │
│ • Initialize DecisionAnalyzer instance                          │
│ • Store in st.session_state.decision_data                       │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ DecisionAnalyzer Instance (in session_state)                    │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Raw Data (pl.LazyFrame)                                 │   │
│ │ ├─ unfiltered_raw_decision_data (always full)           │   │
│ │ └─ decision_data (with global filters applied)          │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Cached Properties (auto-invalidated on filter/level)    │   │
│ │ ├─ sample (10k interactions by default)                 │   │
│ │ ├─ getPreaggregatedFilterView                           │   │
│ │ ├─ getPreaggregatedRemainingView                        │   │
│ │ ├─ color_mappings                                       │   │
│ │ ├─ stages_with_propensity                               │   │
│ │ └─ ... (15+ cached properties)                          │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Query Methods (getDistributionData, getFunnelData, etc) │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Plot Gateway (self.plot → plots.py functions)           │   │
│ └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
          ┌────────────────────────────────────────┐
          │ Analysis Pages (2-11)                  │
          ├────────────────────────────────────────┤
          │ Each page:                             │
          │ 1. Uses ensure_data() guard            │
          │ 2. Calls DecisionAnalyzer methods      │
          │ 3. Applies global filters              │
          │ 4. Caches results via @st.cache_data   │
          │ 5. Displays via st.plotly_chart()      │
          └────────────────────────────────────────┘
```

---

## Key Implementation Patterns

### 1. Data Filtering
```python
# In DecisionAnalyzer
decision_data = apply_filter(unfiltered_raw_decision_data, filters)
applyGlobalDataFilters(filters)

# Filters are Polars expressions combined with AND logic
# Example: (Issue == "Retention") & (Channel == "Email")
```

### 2. Session State Management
```python
# Home.py
st.session_state.decision_data = DecisionAnalyzer(raw_data)
st.session_state.sample_metadata = {"sample_percentage": 50.0, "source_file": "..."}

# Other pages
da = st.session_state.decision_data
```

### 3. Caching Strategy (4 Layers)
```python
# Layer 1: @cached_property on DecisionAnalyzer (auto-invalidated on filter/level)
@cached_property
def sample(self):
    ...

# Layer 2: @st.cache_data on wrapped methods (explicit hash_funcs for Polars)
@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def st_cached_function(data: pl.LazyFrame):
    ...

# Layer 3: Manual cache dicts for thresholding and sensitivity analysis
self._thresholding_cache: dict[tuple, pl.DataFrame]
self._sensitivity_cache: dict[int, pl.LazyFrame]

# Layer 4: Pre-aggregation views (one-time compute, then lazy-wrap)
getPreaggregatedFilterView.collect().lazy()
```

### 4. Interaction-Level Sampling
```python
# Hash-based, deterministic per Interaction ID
sample_rate = min(1.0, target_sample_size / total_interaction_count)
all_rows_for_sampled_interactions = (
    decision_data
    .with_columns((pl.col("Interaction ID").hash() % 1000 < 1000 * sample_rate).alias("_sample"))
    .filter(pl.col("_sample"))
)
```

---

## Known Limitations & Assumptions

1. **Interaction Granularity:** All analyses assume interaction-level data (multiple actions per interaction). Single-interaction filtering would require architectural changes to how sampling and aggregation work.

2. **Sample Size Trade-off:** Larger `sample_size` values improve distribution accuracy but increase computation time. Current default of 10k is tuned for typical use cases.

3. **Stage Level Switching:** When user switches between "Stage Group" and "Stage" levels, all cached properties are invalidated. This can be slow for large datasets.

4. **Pre-aggregation Cost:** Computing `getPreaggregatedFilterView` is expensive and must be done once per data load. Subsequent per-page filtering is fast.

5. **Color Consistency:** Colors are assigned based on full dataset (before filtering) to maintain consistency. This means unused categories still claim colors.

6. **Mandatory Actions:** Mandatory actions bypass normal ranking and always rank first. This is useful for business rules but can obscure ranking patterns.

7. **Propensity Validation:** Code warns if propensities > 1.0 or unusually high (> 10%). Assumes typical marketing propensities are < 1%.

---

## Potential Extension Points

For adding single-interaction filtering, these are the key areas that would need modification:

1. **DecisionAnalyzer.sample** — Currently hash-based sampling over interactions. Would need conditional logic.
2. **Home.py or new page** — Would need UI for interaction ID selection/search.
3. **Cache invalidation** — Single-interaction mode would need to bypass some cached properties.
4. **Page guards** — Each analysis page would need to check if in single-interaction mode and show simplified views.
5. **Plots** — Some distribution plots wouldn't make sense for single interactions.

See [CLAUDE.md](CLAUDE.md) for additional context on naming conventions and code standards.

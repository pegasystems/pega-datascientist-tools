# Impact Analyzer UI Improvements - Design Document

**Date**: 2026-03-09
**Status**: Approved
**Approach**: Shared Utilities with Format Adapters

## Overview

This design brings the Impact Analyzer app's look and feel into alignment with the Decision Analyzer by:

1. **CLI Integration**: Adding `--data-path`, `--sample`, and `--temp-dir` flag support
2. **Home Page Refactoring**: Restructuring data loading to match Decision Analyzer's UX patterns
3. **Analysis Page Styling**: Applying consistent text hierarchy and bordered containers
4. **Code Sharing**: Extracting common patterns into shared utilities to reduce duplication

The approach prioritizes maintainability by creating reusable abstractions while respecting format-specific differences between the two apps.

## Goals

- **Consistency**: Impact Analyzer should feel like part of the same product family as Decision Analyzer
- **Feature Parity**: CLI flags and data loading capabilities should work the same way across both apps
- **Maintainability**: Reduce code duplication by extracting shared patterns
- **User Experience**: Richer feedback, clearer messaging, better visual hierarchy

## Architecture

### Three-Layer Design

**Layer 1: Shared Pipeline** (`streamlit_utils.py`)
- Generic file upload handling (reads bytes, validates extensions)
- CLI path resolution and validation
- Sampling logic (supports `n`, `fraction`, percentage strings)
- Metadata tracking (sample percentage, source file)
- Data summary display templates

**Layer 2: Format Readers** (app-specific utils)
- **Decision Analyzer**: `da_streamlit_utils.py` keeps `load_decision_analyzer()` and format detection
- **Impact Analyzer**: `ia_streamlit_utils.py` provides PDC/VBD readers
- Each returns their native type: DA returns `pl.LazyFrame`, IA returns `ImpactAnalyzer`

**Layer 3: Home Pages** (app-specific)
- Call shared pipeline functions with app-specific readers
- Handle format-specific data summaries
- Store results in session state

### Data Flow for Impact Analyzer

```
1. User uploads/provides data path
   ↓
2. Shared pipeline reads raw bytes/path
   ↓
3. App-specific reader detects format (PDC JSON vs VBD ZIP)
   ↓
4. Shared pipeline applies sampling if --sample flag present
   ↓
5. App-specific reader creates ImpactAnalyzer instance
   ↓
6. Shared pipeline displays summary using app-provided stats
   ↓
7. Store in st.session_state["impact_analyzer"]
```

### Key Design Decisions

1. **Return type flexibility**: Shared functions accept callables that return any type, not just LazyFrame
2. **Sampling point**: Applied to raw data before format-specific processing (consistent with DA)
3. **Metadata storage**: Use same parquet metadata pattern DA uses for sample tracking
4. **Error handling**: Shared functions raise exceptions, apps catch and display user-friendly messages

## CLI Integration

### Supported Flags

**`--data-path <path>`**
- Supports: `.json`, `.ndjson` (PDC), `.zip` (VBD)
- Takes priority over sample data
- Displays error if path doesn't exist or format is unsupported
- Environment variable: `PDSTOOLS_DATA_PATH`

**`--sample <spec>`**
- Formats: `"100000"`, `"100k"`, `"1M"`, `"10%"`
- **Simple random sampling** (no stratification needed for IA)
- Applied to raw data before ImpactAnalyzer instantiation
- Saves sampled data to parquet with metadata
- Environment variable: `PDSTOOLS_SAMPLE_LIMIT`

**`--temp-dir <path>`**
- Directory for sampled parquet files
- Defaults to current working directory
- Environment variable: `PDSTOOLS_TEMP_DIR`

### Loading Priority Chain

```
1. File upload (if present)
   ↓
2. --data-path CLI flag (if set and no upload)
   ↓
3. Sample data (if no upload and no CLI path)
```

### Format Detection for CLI Paths

**PDC (JSON/NDJSON)**:
- Extensions: `.json`, `.ndjson`
- Reader: `ImpactAnalyzer.from_pdc(path)`
- Can be single file or list of files

**VBD (ZIP)**:
- Extension: `.zip`
- Reader: `ImpactAnalyzer.from_vbd(path)`
- Must be single file

### Sampling Strategy: App-Specific Adapters

**Decision Analyzer** (existing):
- **Stratified sampling** by `Interaction ID`
- Keeps all rows for each sampled interaction
- Uses `prepare_and_save()` from `decision_analyzer/utils.py`

**Impact Analyzer** (new):
- **Random row sampling** (no stratification)
- Sample directly from `ia_data` LazyFrame
- New function: `prepare_and_save_random()` in `ia_streamlit_utils.py`

```python
def prepare_and_save_random(
    data: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str = ".",
    source_path: str | None = None,
) -> tuple[pl.LazyFrame, str | None]:
    """Apply random sampling and save to parquet with metadata.

    Performs simple random row sampling (no stratification).
    Writes sampled data to parquet with sample metadata in file metadata.

    Returns:
        (sampled_data, output_path) where output_path is None if no sampling occurred
    """
```

## Home Page Refactoring

### New Structure

The Impact Analyzer Home page will match Decision Analyzer's structure:

**1. Header Section**
```python
standard_page_config(page_title="Impact Analyzer")
show_sidebar_branding("Impact Analyzer")
show_version_header()
```

**2. Page Introduction with Format Table**
```python
"""
# Impact Analyzer

Analyze A/B test experiments from PDC exports or VBD Scenario Planner Actuals.
Two data formats are supported — format is auto-detected on upload:

| | **PDC Export** | **VBD Scenario Planner** |
|---|---|---|
| Format | JSON/NDJSON | ZIP archive |
| Source | Pega Decisioning Center | Value-Based Design |
| Metrics | CTR lift, value lift | Scenario comparison |

All charts are interactive ([Plotly](https://plotly.com/graphing-libraries/)) — pan,
zoom, and hover for details.

### Data import
"""
```

**3. File Upload (Always Visible)**
```python
# Remove dropdown selector
# Show file uploader at top (like DA)
raw_data, uploaded_metadata = handle_file_upload_ia()
```

**4. CLI Path Loading**
```python
configured_path = get_data_path()
configured_path_failed = False
if raw_data is None and configured_path and not has_existing_data:
    with st.spinner(f"Loading data from configured path: {configured_path}"):
        raw_data = handle_data_path_ia()
        data_source_path = configured_path
    if raw_data is not None:
        st.info(f"📂 Loaded data from configured path: `{configured_path}`")
    else:
        configured_path_failed = True
```

**5. Sample Data Fallback**
```python
if not has_new_data and not has_existing_data and not configured_path_failed:
    with st.spinner("Loading sample data"):
        raw_data = handle_sample_data_ia()
    st.info("No file uploaded — using built-in sample data. Upload your own data above to analyze it.")
elif configured_path_failed:
    st.error(f"Failed to load data from configured path: `{configured_path}`. ...")
    st.stop()
```

**6. Sampling & Ingestion**
```python
# Apply --sample flag if present (only for CLI paths, not uploads)
sample_limit_raw = get_sample_limit() if data_source_path else None
if raw_data is not None and sample_limit_raw:
    sample_kwargs = parse_sample_spec(sample_limit_raw)
    with st.spinner("Sampling interactions..."):
        raw_data, sample_path = prepare_and_save_random(...)
    st.info(f"📉 Pre-ingestion sampling applied: keeping **{label}** rows...")
```

**7. Data Summary Display**
```python
def _show_data_summary(ia: ImpactAnalyzer):
    """Display summary banner for loaded Impact Analyzer."""
    # Detect format (PDC vs VBD)
    format_label = "**PDC Export**" or "**VBD Scenario Planner**"

    # Extract statistics from ia.ia_data
    rows = ia.ia_data.select(pl.len()).collect().item()
    channels = ia.ia_data.select(pl.col("Channel").n_unique()).collect().item()
    # Additional stats as appropriate

    summary = (
        f"Data loaded successfully. Detected format: {format_label}\n\n"
        f"**{rows:,}** rows · **{channels}** channels · ..."
    )
    st.success(summary)

    # Show sampling info if applicable
    sample_metadata = st.session_state.get("ia_sample_metadata")
    if sample_metadata:
        sample_pct = sample_metadata["sample_percentage"]
        source_file = sample_metadata.get("source_file", "unknown")
        st.info(f"📊 This data represents **{sample_pct:.1f}%** of the original dataset. Original source: `{source_file}`")
```

### Key Changes from Current IA Home

- **Remove**: Dropdown selector for data source (`"Sample data" | "File upload" | "File path"`)
- **Remove**: Conditional UI based on selection
- **Add**: Always-visible file uploader (matches DA pattern)
- **Add**: CLI path support with priority chain
- **Add**: Rich data summary with row counts, channels, date ranges, etc.
- **Add**: Sample metadata display (when sampling applied)
- **Add**: Capacity error handling (polars index overflow detection)
- **Add**: Time estimates for sampling operations

## Analysis Page Styling

### Consistent Page Pattern

All Impact Analyzer analysis pages will follow Decision Analyzer's structure:

**1. Page Header**
```python
standard_page_config(page_title="Impact Analyzer · Page Name")
ensure_impact_analyzer()  # Guard function

"# Page Title"

"""
Page-level introduction using normal body text.
Explains the purpose and key questions this page answers.
2-4 sentences, high-level orientation.
"""
```

**2. Bordered Containers for Sections**
```python
with st.container(border=True):
    "## Section Title"

    st.caption(
        "Section-level explanation in compact caption style. "
        "Describes what the chart shows and how to interact with it. "
        "1-2 sentences, specific guidance."
    )

    # Chart or content
    st.plotly_chart(fig, width="stretch")
```

**3. Plot Display (Fix Deprecated Parameter)**
```python
# Replace all occurrences:
# OLD: st.plotly_chart(fig, use_container_width=True)
# NEW: st.plotly_chart(fig, width="stretch")
```

### Text Hierarchy Guidelines

**Page-level (triple-quoted strings)**:
- 2-4 sentences
- High-level purpose: "What does this page help me understand?"
- Should orient users immediately after seeing the page title

**Section-level (`st.caption()`)**:
- 1-2 sentences
- Specific guidance: "What does this chart show? How do I use it?"
- Compact and subdued styling (visually de-emphasized)

### Page-by-Page Updates

**`1_Overall_Summary.py`**
- Add page-level intro explaining aggregated lift metrics across channels
- Wrap display options expander in bordered container
- Add section caption explaining the overview chart
- Add section caption explaining the data table
- Fix deprecated `use_container_width=True` → `width="stretch"`

**`2_Trend.py`**
- Add page-level intro about time-series analysis and trends
- Wrap metric selector and chart in bordered container
- Add explanatory captions for chart interactivity
- Fix deprecated parameter

**`3_About.py`**
- Keep minimal (like DA's About page)
- Add version info, links to documentation
- Use bordered container for content sections

## Shared Utilities & Code Organization

### New Shared Functions in `streamlit_utils.py`

**Generic File Upload Handler**
```python
def handle_generic_file_upload(
    label: str,
    supported_types: list[str],
    file_reader_func: Callable[[list], tuple[Any, dict | None]],
    accept_multiple: bool = True,
) -> tuple[Any | None, dict | None]:
    """
    Generic file uploader that delegates format reading to app-specific function.

    Parameters:
        label: Widget label text
        supported_types: List of file extensions (e.g., ["zip", "parquet", "json"])
        file_reader_func: Callable that takes uploaded files and returns (data, metadata)
        accept_multiple: Whether to accept multiple file uploads

    Returns:
        (data, metadata) or (None, None)
    """
```

**Generic CLI Path Handler**
```python
def handle_generic_data_path(
    path_reader_func: Callable[[str], Any],
) -> Any | None:
    """
    Read data from --data-path CLI flag using app-specific reader.

    Handles path validation, zip extraction if needed, error display.

    Parameters:
        path_reader_func: Callable that takes a file path and returns data

    Returns:
        Data object or None if path not configured or loading failed
    """
```

**Sampling Utilities** (moved from `decision_analyzer/utils.py`)
```python
def parse_sample_spec(spec: str) -> dict:
    """
    Parse --sample flag into {n: int} or {fraction: float}.

    Supports: "100000", "100k", "1M", "10%"

    Examples:
        "100000" → {"n": 100000}
        "100k" → {"n": 100000}
        "1M" → {"n": 1000000}
        "10%" → {"fraction": 0.1}
    """

def format_sampling_message(sample_kwargs: dict, row_count: int | None = None) -> str:
    """
    Build user-friendly sampling message with optional time estimates.

    Examples:
        {"n": 100000} → "Sampling 100,000 interactions…"
        {"fraction": 0.1} → "Sampling 10% of the interactions…"
    """
```

### App-Specific Utility Organization

**`da_streamlit_utils.py`** (Decision Analyzer - minimal changes):
- Keep: `load_decision_analyzer()`, `ensure_data()`, stage selectors, filter widgets
- Keep: `handle_file_upload()`, `handle_data_path()`, `handle_sample_data()` as wrappers
- Update: Internally delegate to shared generic handlers where appropriate
- Keep: Stratified sampling logic in `decision_analyzer/utils.py`

**`ia_streamlit_utils.py`** (Impact Analyzer - new additions):
- Keep existing: `load_sample_pdc()`, `load_pdc_from_paths()`, `load_vbd_from_path()`, etc.
- Add: `ensure_impact_analyzer()` with sidebar logo re-apply
- Add: `handle_file_upload_ia()` (wrapper calling shared generic handler)
- Add: `handle_data_path_ia()` (wrapper calling shared generic handler)
- Add: `handle_sample_data_ia()` (loads built-in sample PDC data)
- Add: `prepare_and_save_random()` (random row sampling for IA)
- Add: `_show_data_summary()` (format-aware summary display)

### File Structure After Refactoring

```
python/pdstools/
├── utils/
│   └── streamlit_utils.py
│       ├── Generic upload/path handlers (new)
│       ├── Sampling utilities (moved from DA)
│       ├── Version header, branding (existing)
│       └── Page config, session guards (existing)
│
├── app/
│   ├── decision_analyzer/
│   │   ├── Home.py                 # Minimal changes, uses shared internally
│   │   ├── da_streamlit_utils.py   # Refactored to use shared utilities
│   │   └── pages/*.py              # No changes needed
│   │
│   └── impact_analyzer/
│       ├── Home.py                 # Complete restructure
│       ├── ia_streamlit_utils.py   # New functions for IA format handling
│       └── pages/*.py              # Updated styling, text, containers
│
└── decision_analyzer/
    └── utils.py                    # Keep DA-specific stratified sampling
```

### Backward Compatibility

**Decision Analyzer**: No breaking changes
- Existing functions stay in place
- Internal refactoring to use shared utilities where beneficial
- User-facing behavior unchanged

**Impact Analyzer**: User-facing changes (all improvements)
- Home page UI changes (no dropdown selector, always-visible uploader)
- CLI flags now supported (new feature)
- Richer data summaries (improvement)
- Better error messages and progress feedback (improvement)

## Testing Strategy

### Smoke Tests (Manual)

**Decision Analyzer** (verify no regressions):
1. File upload (various formats) → should work as before
2. `--data-path` with parquet → should work as before
3. `--data-path` with zip → should work as before
4. `--sample 100k` → should work as before
5. Multiple file uploads → should work as before
6. Page navigation and filters → should work as before

**Impact Analyzer** (verify new functionality):
1. File upload PDC JSON → new UI, same data loading
2. File upload VBD ZIP → new UI, same data loading
3. `--data-path` with PDC JSON → new feature
4. `--data-path` with VBD ZIP → new feature
5. `--sample 1000` with PDC → new feature (random sampling)
6. `--sample 10%` with PDC → new feature
7. Page navigation → updated styling
8. Sample data fallback → new UI flow

### Unit Tests (if feasible)

- `parse_sample_spec()` with various formats ("100k", "1M", "10%", invalid inputs)
- Format detection logic for PDC vs VBD
- Metadata reading/writing in parquet files
- Random sampling correctness (row counts, fraction validation)

### Integration Tests

- DA + IA both work in same environment
- CLI flags don't interfere between apps
- Temp directories are app-isolated
- Session state keys are app-namespaced

## Implementation Phases

### Phase 1: Shared Utilities Foundation
1. Create generic upload/path handlers in `streamlit_utils.py`
2. Move sampling utilities from DA to shared location
3. Test with Decision Analyzer (should work identically)

### Phase 2: Impact Analyzer Data Loading
1. Implement `prepare_and_save_random()` for IA sampling
2. Add CLI flag support in IA Home.py
3. Restructure Home.py with new loading flow
4. Add `_show_data_summary()` for IA

### Phase 3: Impact Analyzer Page Styling
1. Update `ensure_impact_analyzer()` helper
2. Apply styling to `1_Overall_Summary.py`
3. Apply styling to `2_Trend.py`
4. Apply styling to `3_About.py`
5. Fix all deprecated `use_container_width` parameters

### Phase 4: Testing & Polish
1. Manual smoke tests on both apps
2. Fix any regressions or issues
3. Update documentation (CLI help, getting started guides)
4. Commit and create PR

## Documentation Updates

Files to update after implementation:

1. **CLI Help Text** (`cli.py`):
   - Update Impact Analyzer description to mention CLI flag support
   - Add note about sampling strategy differences (stratified vs random)

2. **Getting Started Guides**:
   - `GettingStartedWithImpactAnalyzer.rst` (if exists) or create it
   - Document `--data-path`, `--sample`, `--temp-dir` usage
   - Add examples of CLI invocations

3. **CLAUDE.md** (this file):
   - Document Impact Analyzer patterns (if not already covered)
   - Note sampling strategy differences between DA and IA

4. **CHANGELOG** (when released):
   - Note UI improvements and CLI flag support for Impact Analyzer

## Success Criteria

This design will be considered successful when:

1. ✅ Impact Analyzer supports all three CLI flags (`--data-path`, `--sample`, `--temp-dir`)
2. ✅ Impact Analyzer Home page matches Decision Analyzer's UX patterns
3. ✅ Impact Analyzer pages use consistent text hierarchy and containers
4. ✅ No regressions in Decision Analyzer functionality
5. ✅ Code duplication reduced through shared utilities
6. ✅ Both apps pass manual smoke tests
7. ✅ Documentation updated to reflect new capabilities

## Open Questions / Future Considerations

- **Sampling for VBD data**: Current design focuses on PDC (LazyFrame). VBD sampling may need special handling if ZIP contents are not LazyFrame-compatible.
- **Caching for non-parquet sources**: DA caches CSV/JSON to parquet. Should IA do the same? (Likely yes, for consistency)
- **Additional analysis pages**: If IA gets more pages in the future, they should follow the established pattern.
- **Health Check app**: Could benefit from the same refactoring in a future iteration.

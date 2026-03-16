# Claude Code Context for Pega Data Scientist Tools

This file provides context for AI assistants working on this codebase to maintain consistency and follow project conventions.

## Open Source Confidentiality

**CRITICAL:** This is an open source repository. **Never** include customer names, customer-specific data, internal project names, or any identifying information in:
- Code or comments
- Documentation or commit messages
- SPECS files or design docs
- Test data or examples

Use generic descriptions instead (e.g., "sample data" not "Acme Corp data"). This applies to all content that could end up in the public repository.

## Git Workflow

**Never run `git commit` directly on the user's working branch.** Only stage files with `git add` and let the user review and commit manually.

**Exception:** When you have created a dedicated feature/fix branch yourself, or the user explicitly asks you to create a PR, you may commit on that branch.

## Product Naming Conventions

### Decision Analysis Tool
- **User-facing text**: "Decision Analysis Tool"
  - Documentation files (`.rst`, `.md`)
  - Streamlit UI text (headers, captions, markdown)
  - Help text and tooltips
  - Error messages shown to users
- **Code/CLI**: `decision_analyzer`
  - Module names (`pdstools.decision_analyzer`)
  - CLI commands (`pdstools decision_analyzer`)
  - Function and class names
  - File and directory paths

**Example:**
- Docs: "The Decision Analysis Tool provides insights into..."
- CLI: `pdstools decision_analyzer --data-path data.parquet`
- Mixing: "Run the decision_analyzer tool" (should be "Decision Analysis Tool")

### ADM Health Check
- **User-facing text**: "ADM Health Check"
- **Code/CLI**: `adm_healthcheck`

## Documentation Standards

### Version and Upgrade Messages

**Keep update prompts version-focused, not command-focused:**
- Show: version number, "keep up to date" message
- Let dynamic checks show upgrade commands when needed
- Avoid: Hardcoded install commands in static headers

**Rationale:** Users have different package managers (uv, pip, conda). Static commands assume tools that may not be installed.

**Example:**
```python
# Good
st.caption(f"pdstools {version} · Keep up to date")

# Avoid - assumes user has uv
st.caption(f"pdstools {version} · Keep up to date: `uv pip install --upgrade pdstools`")
```

### Analysis Documentation

When documenting analysis types:
- Match current page structure (Overview, Action Funnel, etc.)
- Update descriptions when page names change
- Keep analysis lists focused on key features, not exhaustive
- Highlight V2-specific capabilities (full pipeline vs V1 arbitration-only)

## Tool Preferences

- **Package Manager**: Always use `uv` for Python package management and execution
- **Data Processing**: Prefer Polars over Pandas for data processing. When suggesting Polars syntax, ensure it is valid and tested against the latest Polars version.

## Decision Analyzer Standards

The Decision Analysis Tool (`decision_analyzer`) analyzes Pega decisioning data exports to provide insights into arbitration, filtering, and lever effectiveness.

### File Organization

- **Core library**: `python/pdstools/decision_analyzer/DecisionAnalyzer.py` (main class)
- **Plots module**: `python/pdstools/decision_analyzer/plots.py` (all visualization functions)
- **Streamlit app**: `python/pdstools/app/decision_analyzer/`
  - `Home.py` (entry point)
  - `pages/*.py` (numbered analysis pages)
  - `da_streamlit_utils.py` (shared UI utilities)
- **Data utilities**: `python/pdstools/decision_analyzer/data_read_utils.py` (multi-format ingestion)
- **Schema**: `python/pdstools/decision_analyzer/column_schema.py` (column definitions and display names)

### Data Loading Architecture

**Core library** (`data_read_utils.py`):
- `read_data(path)` — Multi-format reader supporting parquet, csv, json, ndjson, arrow, zip, tar, tar.gz, tgz
- Handles file paths, directories (including Hive-partitioned), and archives
- Automatically extracts zip and tar archives to temp directories
- Calls `_clean_artifacts()` after extraction to remove OS junk files (`__MACOSX`, `._*`, `.DS_Store`)
- Returns `pl.LazyFrame` for lazy evaluation

**Streamlit UI layer** (`da_streamlit_utils.py`):
- `handle_file_upload()` — Multi-file uploader widget supporting all data formats and archives
- `handle_data_path()` — Load from `--data-path` CLI flag with progress feedback
- `handle_sample_data()` — Load built-in demo data (EEV2 sample)
- `_read_uploaded_zip()`, `_read_uploaded_tar()` — Archive extraction with time estimation spinners
- All functions use `read_data()` internally for consistency

**CLI integration** (`cli.py`):
- `--data-path` → `PDSTOOLS_DATA_PATH` env var → `get_data_path()` — Specify source file/directory
- `--sample` → `PDSTOOLS_SAMPLE_LIMIT` env var → `get_sample_limit()` — Pre-ingestion sampling (e.g., "100k" or "10%")
- `--temp-dir` → `PDSTOOLS_TEMP_DIR` env var → `get_temp_dir()` — Directory for sampled/cached files

**Sampling flow:**
1. CLI provides `--sample` flag (e.g., "100k" or "10%")
2. `parse_sample_flag()` converts to `{n: 100000}` or `{fraction: 0.1}`
3. `prepare_and_save()` calls `sample_interactions()` for deterministic hash-based sampling
4. Sampled data written as parquet with metadata tracking source file and sample percentage
5. If data is smaller than requested sample size, file writing is skipped and full data is used

⚠️ **Concurrent access**: Running multiple Decision Analyzer instances with overlapping `--temp-dir` or `--data-path` pointing to the same files can cause race conditions. Use separate temp directories or avoid sharing sampled files between instances.

### Design Patterns

**Propensity Display**: All propensity values must be displayed as percentages (e.g., "12.345%"), not raw decimals. This applies to:
- Chart axes, labels, and hover text
- Table cells and headers
- Slider values and input fields
- Component distributions (violin plots, ECDFs, box plots)

**Stage Level Selection**: Users can toggle between "Stage Group" and "Stage" granularity. Use the shared `stage_level_selector()` helper from `da_streamlit_utils.py` for consistent UI. Store the selection in `st.session_state.stage_level`.

**Shared UI Components**: Use utilities from `da_streamlit_utils.py`:
- `stage_selectbox()` — stage selector with stage-group-aware formatting
- `stage_level_selector()` — granularity toggle
- `ensure_data()` — guard for pages requiring loaded data

**Session State**: Minimize what's stored in `st.session_state`. Prefer computing values on-demand from the `DecisionAnalyzer` instance rather than caching intermediate results.

**Dynamic Stages**: Avoid hardcoding stage names (e.g., "Final", "Arbitration"). Get stage lists from data and use display name mappings where available.

**Mandatory Actions**: Mandatory actions use special arbitration priority values (4999999 or 4999999999) that bypass normal prioritization. The `DecisionAnalyzer` accepts a `mandatory_expr` parameter to tag these at initialization. When implementing features involving ranking or sensitivity analysis, consider how mandatory actions should be handled (typically: flag them clearly, exclude from sensitivity analyses).

### Code Organization

**Delegation Principle**: Keep Streamlit page code focused on UI presentation. Delegate data processing to `DecisionAnalyzer` methods and visualization to `plots.py` functions. Avoid complex business logic in page files.

**Plot Functions**: All visualization functions belong in `plots.py`. Page files should only configure and display plots, not define them inline.

### Streamlit Text Styling

**Explanatory Text Hierarchy**: Use consistent text styling to create clear visual hierarchy:

- **Page-level introduction** (immediately after page title): Use triple-quoted strings `"""..."""` for normal body text
  - Purpose: High-level page description and key questions
  - Should remain prominent to orient users

- **Section-level explanations** (inside `st.container(border=True)` blocks): Use `st.caption()` for smaller, subdued text
  - Purpose: Describe specific charts, tables, or interactions
  - Should be concise and visually de-emphasized relative to the content

**Example:**
```python
"# Page Title"

"""
Page-level introduction explaining the purpose and key questions.
This uses normal body text for emphasis.
"""

ensure_data()

with st.container(border=True):
    "## Section Title"

    st.caption(
        "Section explanation describing the chart below. "
        "This uses caption styling for a more compact appearance."
    )

    st.plotly_chart(fig)
```

**Writing Guidelines:**
- Keep captions concise (aim for 30-50 words per section)
- Preserve markdown formatting (bold, italics) within captions
- Explain what the visualization shows and how to use it
- Remove redundant information that's obvious from context

## Testing Expectations

### Test Coverage Requirements

**Minimum coverage: 80%** for all pdstools code. CI/CD enforces this requirement.

- **Patch coverage**: New code in PRs must have ≥80% coverage
- **Overall coverage**: Entire codebase must maintain ≥80% coverage
- **When adding features**: Write tests before or alongside implementation
- **When fixing bugs**: Add regression tests that fail before the fix and pass after

Check coverage locally (requires `pytest-cov`):
```bash
uv run pytest python/tests --cov=python/pdstools --cov-report=term-missing
```

### Before Committing
- Run pytest: `uv run pytest python/tests`
- Verify all tests pass, including new tests for your changes
- Build Sphinx docs: `cd python/docs && make html`
- Smoke test apps after UI changes: `pdstools decision_analyzer`

### Documentation Changes
- Verify Sphinx builds without warnings
- Check rendered HTML for formatting issues
- Spell-check modified sections

## File Organization

### Documentation
- Getting started guides: `python/docs/source/GettingStartedWith*.rst`
- Implementation plans: `docs/plans/YYYY-MM-DD-<topic>.md`
- Design docs: `docs/plans/YYYY-MM-DD-<topic>-design.md`

### Streamlit Apps
- Home pages: `python/pdstools/app/*/Home.py`
- Analysis pages: `python/pdstools/app/*/pages/*.py`
- Shared utilities: `python/pdstools/utils/streamlit_utils.py`

### Quarto Reports
- Main reports: `python/pdstools/reports/HealthCheck.qmd`, `python/pdstools/reports/ModelReport.qmd`
- GlobalExplanations templates: `python/pdstools/reports/GlobalExplanations/assets/templates/*.qmd`
- Report utilities: `python/pdstools/utils/report_utils.py`
- Formatting config: `python/pdstools/utils/metric_limits.py`, `python/pdstools/utils/number_format.py`

## Quarto Report Standards

Quarto reports provide static, portable HTML analysis documents for ADM data. The primary reports are the ADM Health Check and the Model Report. GlobalExplanations reports use a simpler template-based approach.

### Design Principles

**Delegation over Duplication**: Reports should be lightweight orchestrators, not implementations.
- Use `pdstools` methods for data processing (e.g., `datamart.aggregates.summary_by_channel()`)
- Use `pdstools.plot.*` methods for visualizations
- Keep code cells focused on configuration and presentation, not complex logic
- Avoid duplicating business logic that belongs in the library

**Target Audience**: Technical business analysts and data scientists who understand ADM concepts but need concrete interpretation guidance.

**Consistency Elements**:
- Pega logo and branding via `assets/pega-report-overrides.css`
- Credits section at end using `report_utils.show_credits()`
- Version information in collapsible callout at end
- Standard error handling with `report_utils.quarto_plot_exception()`
- Consistent YAML front matter (title-block-banner, Pega theme, etc.)

### Formatting Standards for Rates and Propensities

**Critical Rule**: All rates, propensities, CTR, and similar percentage values **must be displayed as percentages with exactly 3 decimal places** (e.g., "12.345%").

Use the centralized `MetricFormats` system defined in `python/pdstools/utils/metric_limits.py`:

```python
# In create_metric_gttable calls, specify the metric ID for RAG coloring:
column_to_metric={
    "CTR": report_utils.exclusive_0_1_range_rag,  # Auto-formats as X.XXX%
    "Performance": "ModelPerformance",  # Auto-formats as XX (scaled to 0-100)
    "Base Propensity": report_utils.exclusive_0_1_range_rag,  # Auto-formats as X.XXX%
}
```

**Do NOT** format rates/propensities manually in Polars unless necessary for intermediate calculations. The table rendering functions (`create_metric_gttable`, `create_metric_itable`) automatically apply the correct format from `MetricFormats` when you specify the metric ID or RAG function.

**Common Metrics**:
- `CTR`: 3 decimals, percentage (12.345%)
- `ModelPerformance`: 2 decimals, scaled to 0-100 (87.50)
- `EngagementLift`: 0 decimals, percentage (125%)
- `OmniChannelPercentage`: 1 decimal, percentage (45.2%)

**Adding New Metrics**: Register format in `MetricFormats._FORMATS` in `metric_limits.py`:

```python
"MyMetric": NumberFormat(decimals=3, scale_by=100, suffix="%"),
```

### Visual Consistency

**Plots**:
- Use `template="pega"` for Plotly figures
- Clear, descriptive titles (use `<br><sup>subtitle</sup>` for additional context)
- Proper axis labels (not empty strings or generic labels)
- Include guidance callouts for interpretation where needed
- Consistent color schemes (use Pega colors where applicable)

**Tables**:
- Use `report_utils.create_metric_gttable()` for all tables containing metrics
- Apply RAG coloring via `column_to_metric` parameter
- Hide technical columns (e.g., `isValid`, `usesNBAD`) with `.cols_hide()`
- Use `.cols_label()` for user-friendly column names (e.g., "CTR" → "Base Rate")
- Use `.tab_spanner()` to group related columns with headers

### Structure

Standard report sections:
1. **Title and metadata** (automatically from YAML front matter)
2. **Overview** with key metrics and global view
3. **Detailed analysis sections** with specific insights
4. **Guidance callouts** with interpretation tips using Quarto callouts
5. **Credits section** using `report_utils.show_credits("path/to/report.qmd")`
6. **Version information** in collapsible callout with `show_versions()`

### Common Patterns

**Exception Handling**:
```python
try:
    fig = datamart.plot.score_distribution(model_id=model_id)
    fig.update_layout(title="Distribution", xaxis_title="Score")
    fig.show()
except Exception as e:
    report_utils.quarto_plot_exception("Score Distribution", e)
```

**Conditional Content**:
```python
if condition:
    report_utils.quarto_print("Additional analysis...")
```

**Interactive Plot Disclaimer** (standard callout):
```markdown
::: {.callout-tip}
Charts are built with [Plotly](https://plotly.com/python/) and have [user controls for panning, zooming etc](https://plotly.com/chart-studio-help/zoom-pan-hover-controls/).
These interactive plots do not render well in portals like Sharepoint or Box. View from a browser for best experience.
:::
```

**Table RAG Coloring**:
```python
formatted_table = report_utils.create_metric_gttable(
    df,
    title="Overview",
    column_to_metric={
        "Performance": "ModelPerformance",
        "CTR": report_utils.exclusive_0_1_range_rag,
        ("Actions", "Used Actions"): "ActionCount",  # Multiple columns same metric
    },
)
display(formatted_table.cols_hide(["technical_column"]))
```

### Notes on GlobalExplanations Reports

The GlobalExplanations reports (`python/pdstools/reports/GlobalExplanations/`) use a simpler template-based approach with string substitution (e.g., `{TOP_N}`, `{ROOT_DIR}`). These are composable templates rather than standalone reports and may not follow all the conventions above. Focus consistency efforts on HealthCheck.qmd and ModelReport.qmd as the reference implementations.

## Code Patterns

### Streamlit Page Structure
```python
standard_page_config(page_title="Tool Name · Page Name")
show_sidebar_branding("Tool Name")

"# Page Title"
"""
Page description...
"""

ensure_data()  # Guard for pages requiring loaded data
```

### Streamlit Chart Display

Do not pass `width="stretch"` or other extra keyword arguments to `st.plotly_chart()`.
Streamlit 1.50 deprecated `**kwargs` on `plotly_chart`; any unrecognized keyword triggers
a deprecation warning. Since `use_container_width=True` is the default, omit it for
full-width charts. Use `use_container_width=False` only when you need content-width.

For Plotly configuration options (e.g., disabling scroll zoom), use the explicit `config` parameter.

**Example**:
```python
# Good — full width (default)
st.plotly_chart(fig)

# Good — content width
st.plotly_chart(fig, use_container_width=False)

# Good — Plotly config
st.plotly_chart(fig, config={"scrollZoom": False})

# Avoid — triggers deprecation warning
st.plotly_chart(fig, width="stretch")
```

### Streamlit Color Consistency

**Pattern: Pre-compute categorical color mappings for session consistency**

When using categorical dimensions for plot colors in Streamlit apps, colors must remain consistent throughout the session regardless of filtering. Dynamic color assignment (based on filtered data) causes confusion when legends change colors as users interact with controls.

**Implementation:**
- Use `pdstools.utils.color_mapping.create_categorical_color_mappings()` utility
- Compute mappings once at data load time (not from filtered subsets)
- Cache with `@cached_property` or `@st.cache_resource`
- Pass to Plotly via `color_discrete_map` parameter

**Example:**
```python
from pdstools.utils.color_mapping import create_categorical_color_mappings
from pdstools.utils.pega_template import colorway

@cached_property
def color_mappings(self):
    return create_categorical_color_mappings(
        self.data,
        ["Category", "Type", "Status"],
        colorway,
        column_mapping=self.column_mapping  # optional
    )

# In plot functions:
fig = px.bar(
    filtered_data,
    x="x",
    y="y",
    color="Category",
    color_discrete_map=self.color_mappings.get("Category")
)
```

**Rationale:**
- Users can track categories across different views/filters
- Deterministic (alphabetically sorted values → consistent colors)
- Minimal performance overhead (computed once, cached)
- Plotly only colors visible values, but uses consistent assignments

**User Communication:**
Add an informational message on filter pages explaining that colors remain consistent:
```python
st.info(
    "Note: Chart colors remain consistent throughout your session "
    "based on all values in the loaded dataset, even when filters reduce visible data."
)
```

## Plot Function Patterns

### Standard Plot Function Signature

All plot functions in `plots.py` should follow these conventions:

```python
def my_plot(
    self,
    stage: str,                                    # Required parameters first
    scope: str = "Action",                         # Common optional parameters
    color_by: str | None = None,                   # Optional coloring dimension
    additional_filters: pl.Expr | list[pl.Expr] | None = None,
    return_df: bool = False,                       # Always include for testing
) -> go.Figure | pl.LazyFrame:
    """Clear description of what the plot shows.

    Parameters
    ----------
    stage : str
        The stage to analyze
    scope : str, default "Action"
        Granularity level (Issue, Group, or Action)
    color_by : str, optional
        Categorical dimension for coloring (e.g., "Channel", "Stage")
    additional_filters : pl.Expr or list of pl.Expr, optional
        Extra filters to apply before aggregation
    return_df : bool, default False
        If True, return the data instead of the figure (for testing)

    Returns
    -------
    go.Figure or pl.LazyFrame
        Plotly figure or the underlying data if return_df=True
    """
    # Get data from DecisionAnalyzer method (delegate business logic)
    df = self._decision_data.get_plot_data(stage=stage, scope=scope, ...)

    if return_df:
        return df

    # Use consistent colors if color_by specified
    color_discrete_map = None
    if color_by is not None:
        color_discrete_map = self._decision_data.color_mappings.get(color_by)

    # Create figure
    fig = px.bar(
        df.collect(),
        x="x",
        y=scope,  # Use scope parameter for display
        color=color_by,
        color_discrete_map=color_discrete_map,
        template="pega",
    )

    return fig
```

### Key Principles

**Scope/Granularity Parameter**: When plots display categorical hierarchies (Issue → Group → Action), accept a `scope` parameter:
```python
def component_drilldown(self, component_name: str, scope: str = "Action", ...):
    # Use scope for both:
    # 1. y-axis display
    fig.add_trace(go.Bar(y=plot_df[scope], ...))

    # 2. grouping in underlying data queries
    df = self._decision_data.getComponentDrilldown(...)
```

**Color By Parameter**: For plots that can show breakdowns by different dimensions:
```python
def action_variation(self, stage: str, color_by: str | None = None, ...):
    # Special handling for combined dimensions like "Channel/Direction"
    if color_by == "Channel/Direction":
        # Create synthetic column in data processing
        distribution = distribution.with_columns(
            pl.concat_str("Channel", "Direction", separator="/").alias("Channel/Direction")
        )
```

**Return DataFrame for Testing**: Always include `return_df` parameter to enable data validation tests:
```python
def test_return_df(self, plot_instance):
    df = plot_instance.my_plot(stage="Output", return_df=True)
    assert isinstance(df, pl.LazyFrame)
    assert "expected_column" in df.collect_schema().names()
```

**Consistent Color Mapping**: Always check for and use cached color mappings:
```python
color_discrete_map = None
if color_by is not None:
    color_discrete_map = self._decision_data.color_mappings.get(color_by)

fig = px.bar(..., color=color_by, color_discrete_map=color_discrete_map)
```

### Streamlit Wrapper Functions

When plot functions need Streamlit caching, create thin wrappers in `da_streamlit_utils.py`:

```python
@st.cache_data(hash_funcs=polars_lazyframe_hashing)
def st_my_plot(data: pl.LazyFrame, stage: str, color_discrete_map: dict | None = None):
    """Cached wrapper for my_plot function.

    Note: Pass color_discrete_map explicitly since @cached_property
    objects are not hashable by Streamlit.
    """
    return plot_my_plot(data, stage, color_discrete_map=color_discrete_map)
```

In page files:
```python
# Get color mapping from cached property
color_map = st.session_state.decision_data.color_mappings.get("Channel")

# Call cached wrapper with explicit parameters
fig = st_my_plot(
    filtered_data,
    stage="Arbitration",
    color_discrete_map=color_map
)
st.plotly_chart(fig)
```

### Combined Dimensions (e.g., Channel/Direction)

When users need to see data broken down by combinations like "Channel/Direction":

**In data processing functions:**
```python
def get_data(self, color_by: str | None = None):
    # Detect combined dimension request
    needs_channel_direction = color_by == "Channel/Direction"

    if needs_channel_direction:
        # Group by individual columns
        grouping_levels = ["Channel", "Direction", "Action"]
    else:
        grouping_levels = [color_by, "Action"] if color_by else ["Action"]

    # Get distribution
    df = self.getDistributionData(grouping_levels=grouping_levels).collect()

    # Create synthetic combined column
    if needs_channel_direction:
        df = df.with_columns(
            pl.concat_str("Channel", "Direction", separator="/").alias("Channel/Direction")
        )
```

**In color mapping initialization:**
```python
# Channel/Direction is a special combined dimension
categorical_columns = [
    "Issue", "Group", "Action", "Channel", "Direction",
    "Channel/Direction",  # Synthetic but needs consistent colors
]

# The color_mapping utility handles the "/" by creating mappings
# from the full dataset's Channel × Direction combinations
```

### Testing Plot Functions

Standard test structure for plot functions:

```python
class TestMyPlot:
    """Test my_plot method."""

    def test_basic_plot(self, plot_instance):
        """Test plot creation with default parameters."""
        fig = plot_instance.my_plot(stage="Output")
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_instance):
        """Test dataframe return for data validation."""
        df = plot_instance.my_plot(stage="Output", return_df=True)
        assert isinstance(df, pl.LazyFrame)
        # Verify data structure
        schema = df.collect_schema().names()
        assert "expected_column" in schema

    def test_with_color_by(self, plot_instance):
        """Test plot with color dimension."""
        fig = plot_instance.my_plot(stage="Output", color_by="Channel")
        assert isinstance(fig, Figure)
        # Verify legend title matches color dimension
        assert fig.layout.legend.title.text == "Channel"

    def test_with_scope(self, plot_instance):
        """Test plot respects scope parameter."""
        for scope in ["Issue", "Group", "Action"]:
            fig = plot_instance.my_plot(stage="Output", scope=scope)
            assert isinstance(fig, Figure)
            # Verify y-axis reflects the scope
```

### Documentation Formatting
- Use relative links for internal docs
- Code blocks: use appropriate language tags
- CLI examples: show with optional `[uv run]` prefix for both workflows

## Code Quality Standards

Before presenting code for commit, review it as a senior developer would:

### Comments and Documentation
- **No redundant comments**: Comments should explain *why*, never *what*. Remove comments that merely restate the code.
- **Single source of truth for docs**: Do not duplicate documentation between a public method/property and its private implementation. Keep the canonical description in one place and reference it from the other.
- **Clean docstrings**: Every public method/property should have a concise, accurate docstring. Internal methods need only a brief summary unless the logic is non-obvious.

### Code Cleanliness
- **DRY (Don't Repeat Yourself)**: Eliminate dead code, unused variables, and duplicated logic. If a value is computed but never used, remove it.
- **Minimal diffs**: Only change what is necessary. Do not reformat unrelated code or touch files outside the scope of the task.
- **Consistent naming**: Follow the project's naming conventions (snake_case for functions/variables, PascalCase for classes). Avoid misleading names.

### Testing
- **Test accuracy**: Module-level docstrings and test names must accurately describe what is being tested.

## Python Type Hints

This project supports Python 3.10+. Use modern type hint syntax:
- Use `X | None` instead of `Optional[X]`
- Use `X | Y` instead of `Union[X, Y]`
- Use `list[str]` instead of `List[str]`
- Use `dict[str, X]` instead of `Dict[str, X]`

**Do not** import `Optional`, `Union`, `List`, `Dict` from `typing` — use built-in generics and `|` unions.

## Logging Standards

### When to Use Logging vs Debug Parameters

PDS Tools uses Python's `logging` module for diagnostic output. Use logging for:
- Progress messages during long-running operations
- Diagnostic information about data transformations
- Error messages in library code
- Warnings about deprecated features or unexpected data

**Do NOT use print statements in library code.** Print statements are only acceptable in:
- CLI code for user-facing output (prompts, menus, results)
- Interactive notebook examples
- Test debugging (temporary only)

### Parameter Naming Conventions

**`debug` parameter**: Use when the parameter changes the **return value** (e.g., returns additional columns, more detailed data structure). This is NOT for logging.

Example:
```python
def summary_by_channel(self, ..., debug: bool = False) -> pl.LazyFrame:
    """
    ...
    debug : bool, default False
        If True, include extra columns (ModelTechnique, Configurations) in output.
    """
    result = result.group_by(...).agg(...)
    if debug:
        return result  # Keep all columns
    return result.drop("ModelTechnique", "Configurations")  # Hide internal columns
```

**`verbose` parameter**: REMOVED. Never use this pattern. If you find a `verbose` parameter in code:
1. Remove the parameter from the function signature completely
2. Replace `if verbose: print(...)` with unconditional `logger.debug(...)`
3. Update docstrings to remove the parameter and mention logging configuration
4. Remove verbose from all function calls in the codebase

### Logging Pattern

Every module should have a module-level logger:

```python
import logging

logger = logging.getLogger(__name__)

class MyClass:
    def my_method(self, data):
        logger.debug(f"Processing {len(data)} records")
        try:
            result = self._compute(data)
            logger.debug(f"Computation complete: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Failed to compute: {e}")
            raise
```

### Testing Logging

Test logging output using pytest's `caplog` fixture:

```python
import logging

def test_my_method_logging(caplog):
    with caplog.at_level(logging.DEBUG):
        obj = MyClass()
        obj.my_method(data)

    assert "Processing" in caplog.text
    assert any(r.levelname == "DEBUG" for r in caplog.records)
```

## Type Checker Noise

When the IDE reports type errors after a file save, distinguish between **pre-existing** issues (already present before the edit) and **newly introduced** issues. For pre-existing type errors, ask the user whether to include a fix in the current change rather than silently fixing or ignoring them. This avoids scope creep while still surfacing opportunities.

## Jupyter Notebooks

When suggesting code changes to Python notebooks, **do not try to edit the notebook file directly**. Instead, provide only the code that should be in the cell, and let the user paste it into the notebook.

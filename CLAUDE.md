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

## Testing Expectations

### Before Committing
- Run pytest: `uv run pytest python/tests`
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

**Deprecated parameter**: Do not use `use_container_width` in `st.plotly_chart()` calls. This parameter was deprecated after 2025-12-31.

**Use these instead**:
- `width="stretch"` — equivalent to `use_container_width=True`
- `width="content"` — equivalent to `use_container_width=False`

**Example**:
```python
# Good
st.plotly_chart(fig, width="stretch")

# Avoid (deprecated)
st.plotly_chart(fig, use_container_width=True)
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

## Type Checker Noise

When the IDE reports type errors after a file save, distinguish between **pre-existing** issues (already present before the edit) and **newly introduced** issues. For pre-existing type errors, ask the user whether to include a fix in the current change rather than silently fixing or ignoring them. This avoids scope creep while still surfacing opportunities.

## Jupyter Notebooks

When suggesting code changes to Python notebooks, **do not try to edit the notebook file directly**. Instead, provide only the code that should be in the cell, and let the user paste it into the notebook.

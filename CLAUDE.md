# Claude Code Context for Pega Data Scientist Tools

This file provides context for AI assistants working on this codebase to maintain consistency and follow project conventions.

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

## Testing Expectations

### Before Committing
- Run pytest: `pytest python/tests`
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

### Documentation Formatting
- Use relative links for internal docs
- Code blocks: use appropriate language tags
- CLI examples: show with optional `[uv run]` prefix for both workflows

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

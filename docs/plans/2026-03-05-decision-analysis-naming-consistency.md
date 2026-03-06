# Decision Analysis Tool Naming Consistency

**Date:** 2026-03-05
**Status:** Approved
**Type:** Documentation + Style Guide

## Overview

Standardize the naming of the Decision Analyzer tool to "Decision Analysis Tool" in all user-facing text while maintaining "decision_analyzer" in code, CLI commands, and internal references. Create style guide documentation to ensure future consistency.

## Background

The tool is currently referred to inconsistently across documentation and UI:
- "Decision Analyzer Tool"
- "Decision Analyzer"
- "The Decision Analyzer"
- "Decision Analysis" (in some places)

This inconsistency creates confusion for users. We need to standardize on a single user-facing name.

## Goals

1. **Standardize naming** - Use "Decision Analysis Tool" consistently in all user-facing text
2. **Update documentation** - Fix naming in getting started guide and analysis types list
3. **Clean up UI text** - Remove redundant "uv pip install" helper text from version header
4. **Create style guide** - Document naming conventions for future consistency
5. **Add CLAUDE.md** - Provide AI assistant context for maintaining consistency

## Design

### 1. Naming Convention

**User-facing text (change to "Decision Analysis Tool"):**
- Documentation files (`.rst`, `.md`)
- Streamlit UI text (headers, captions, markdown strings)
- Help text and tooltips
- Error messages shown to users
- Page titles and descriptions

**Code/internal (keep as "decision_analyzer"):**
- Python module names (`pdstools.decision_analyzer`)
- CLI commands (`pdstools decision_analyzer`)
- Function and class names
- File and directory paths
- Internal code references

### 2. File Updates

#### Documentation Updates

**File:** `python/docs/source/GettingStartedWithDecisionAnalyzer.rst`

Changes:
- Title: "Getting Started with the Decision Analyzer Tool" → Keep (already correct)
- "What is the Decision Analyzer Tool?" → Keep (already correct)
- Body text: Update "The Decision Analyzer" → "The Decision Analysis Tool"
- Analysis types list: Update to reflect current V2-focused pages
  - Emphasize: Overview, Action Funnel (full pipeline), Optionality Analysis
  - Keep: Action Distribution, Global Sensitivity, Win/Loss, Arbitration Component Distribution
  - Remove: "Lever Experimentation" (not a separate page)
  - Update descriptions to match current page names

**File:** `python/pdstools/utils/streamlit_utils.py`

Line 114 - Current:
```python
st.caption(
    f"pdstools {pdstools_version} · Keep up to date: `uv pip install --upgrade pdstools`",
)
```

Change to:
```python
st.caption(
    f"pdstools {pdstools_version} · Keep up to date",
)
```

Keep lines 123 and 646 (upgrade warning messages with the uv command).

**File:** `python/pdstools/app/decision_analyzer/Home.py`

Line 34 - Current:
```python
# Decision Analysis
```

Keep as-is (already correct).

Review markdown strings for any "Decision Analyzer" references.

#### Style Guide Creation

**File:** `CLAUDE.md` (new file at repository root)

Purpose: AI assistant context for maintaining codebase consistency

Content:
- Product naming conventions
- Documentation patterns
- Code organization principles
- Testing requirements

**File:** `CONTRIBUTING.md` (update existing)

Add new section: "Style Guide"

Content:
- Product naming conventions (detailed)
- Documentation standards
- UI text guidelines
- Examples with before/after

### 3. Systematic Update Process

**Discovery:**
```bash
# Find all user-facing instances
grep -r "Decision Analyzer" --include="*.rst" --include="*.md" python/docs/
grep -r '"# Decision Analyzer\|"Decision Analyzer' --include="*.py" python/pdstools/app/
```

**Classification:**
For each instance found, determine:
1. Is it user-facing? → Change to "Decision Analysis Tool"
2. Is it code/CLI? → Keep as "decision_analyzer"
3. Is it a path reference? → Keep as-is

**Update pattern:**
- "The Decision Analyzer" → "The Decision Analysis Tool"
- "Decision Analyzer Tool" → "Decision Analysis Tool"
- "Using the Decision Analyzer" → "Using the Decision Analysis Tool"
- Keep: `decision_analyzer` (code), `pdstools decision_analyzer` (CLI)

### 4. Documentation Content

#### CLAUDE.md Structure

```markdown
# Claude Code Context for Pega Data Scientist Tools

## Product Naming Conventions

### Decision Analysis Tool
- User-facing: "Decision Analysis Tool"
- Code/CLI: `decision_analyzer`

### ADM Health Check
- User-facing: "ADM Health Check"
- Code/CLI: `adm_healthcheck`

## Documentation Standards

- Keep upgrade prompts version-focused, not command-focused
- Analysis types lists should match current page structure
- Spell-check all user-facing documentation

## Testing

- Run pytest before committing
- Build Sphinx docs to verify changes
- Smoke test apps after UI text changes
```

#### CONTRIBUTING.md Addition

```markdown
## Style Guide

### Product Naming

When referring to pdstools applications:

| Tool | User-facing name | Code/CLI reference |
|------|------------------|-------------------|
| Decision Analyzer | Decision Analysis Tool | `decision_analyzer` |
| ADM Health Check | ADM Health Check | `adm_healthcheck` |

**Rules:**
- Documentation, UI headers, help text → Use user-facing name
- Code, CLI commands, file paths → Use code reference
- Be consistent within each context

### Version and Upgrade Text

**Do:**
- Show current version
- Show "Keep up to date" message
- Let dynamic checks show specific upgrade commands when needed

**Don't:**
- Include specific install commands in static headers
- Assume user's package manager (uv vs pip vs conda)

### Examples

**Good:**
```python
st.caption(f"pdstools {version} · Keep up to date")
```

**Avoid:**
```python
st.caption(f"pdstools {version} · Keep up to date: `uv pip install --upgrade pdstools`")
```
```

### 5. Validation Strategy

**Step 1: Grep validation**
```bash
# Should find no user-facing instances (only code references)
grep -r "Decision Analyzer" --include="*.rst" --include="*.md" python/docs/
# Should find only markdown headers and strings, not identifiers
grep -r '"Decision Analyz' --include="*.py" python/pdstools/app/
```

**Step 2: Documentation build**
```bash
cd python/docs && make html
# Check for warnings or errors
# Verify updated text renders correctly
```

**Step 3: Spell check**
- Review GettingStartedWithDecisionAnalyzer.rst
- Check for typos in modified sections
- Verify analysis descriptions are accurate

**Step 4: App smoke test**
```bash
pdstools decision_analyzer
```
- Verify Home page naming
- Check version header (should not show uv pip install)
- Confirm pages match documentation descriptions

## Success Criteria

- [ ] All user-facing text uses "Decision Analysis Tool"
- [ ] All code/CLI uses "decision_analyzer"
- [ ] Documentation builds without errors
- [ ] Getting started guide analysis types match current pages
- [ ] Version header doesn't include install command
- [ ] CLAUDE.md exists and documents conventions
- [ ] CONTRIBUTING.md includes style guide section
- [ ] Spell check complete on modified docs
- [ ] App launches and displays correct naming

## Future Considerations

- Apply same naming conventions to other pdstools apps
- Consider adding pre-commit hook for naming consistency checks
- Update any external documentation (wiki, blog posts) if applicable
- Review other tools (ADM Health Check, etc.) for similar consistency issues

## References

- Getting Started docs: `python/docs/source/GettingStartedWithDecisionAnalyzer.rst`
- Streamlit utils: `python/pdstools/utils/streamlit_utils.py`
- Home page: `python/pdstools/app/decision_analyzer/Home.py`
- Contributing guide: `CONTRIBUTING.md`

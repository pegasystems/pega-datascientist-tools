# Decision Analysis Tool Naming Consistency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize "Decision Analysis Tool" in user-facing text, create style guide for future consistency.

**Architecture:** Systematic grep-based discovery, manual classification of user-facing vs code references, documentation updates with style guide creation.

**Tech Stack:** Sphinx (documentation), Streamlit (UI), grep/sed (text processing)

---

## Task 1: Create CLAUDE.md style guide

**Files:**
- Create: `CLAUDE.md`

**Step 1: Create CLAUDE.md with naming conventions**

```markdown
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
- ✅ Docs: "The Decision Analysis Tool provides insights into..."
- ✅ CLI: `pdstools decision_analyzer --data-path data.parquet`
- ❌ Mixing: "Run the decision_analyzer tool" (should be "Decision Analysis Tool")

### ADM Health Check
- **User-facing text**: "ADM Health Check"
- **Code/CLI**: `adm_healthcheck`

## Documentation Standards

### Version and Upgrade Messages

**Keep update prompts version-focused, not command-focused:**
- ✅ Show: version number, "keep up to date" message
- ✅ Let dynamic checks show upgrade commands when needed
- ❌ Avoid: Hardcoded install commands in static headers

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
```

**Step 2: Verify file was created**

Run: `cat CLAUDE.md | head -20`
Expected: See header and naming conventions section

**Step 3: Commit CLAUDE.md**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md style guide for AI assistants

Provides naming conventions, documentation standards, and testing
expectations for maintaining consistency across the codebase.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update CONTRIBUTING.md with style guide

**Files:**
- Modify: `CONTRIBUTING.md`

**Step 1: Add style guide section to CONTRIBUTING.md**

Add after the "Documentation and articles" section:

```markdown
## Style Guide

### Product Naming Conventions

When referring to pdstools applications, use consistent naming:

| Tool | User-facing name | Code/CLI reference |
|------|------------------|-------------------|
| Decision Analyzer | Decision Analysis Tool | `decision_analyzer` |
| ADM Health Check | ADM Health Check | `adm_healthcheck` |

**Rules:**
- **Documentation, UI headers, help text** → Use user-facing name
- **Code, CLI commands, file paths** → Use code reference
- Be consistent within each context

**Examples:**

✅ **Good:**
```markdown
# Documentation
The Decision Analysis Tool provides comprehensive insights...

# Code
from pdstools.decision_analyzer import DecisionAnalyzer
```

❌ **Avoid:**
```markdown
# Don't mix contexts
Run the decision_analyzer tool to analyze decisions.
# Should be: "Run the Decision Analysis Tool..."
```

### Version and Upgrade Text

**Do:**
- Show current version number
- Show "Keep up to date" message
- Let dynamic version checks show specific upgrade commands when needed

**Don't:**
- Include specific install commands in static UI headers
- Assume user's package manager (uv vs pip vs conda)

**Examples:**

✅ **Good:**
```python
st.caption(f"pdstools {version} · Keep up to date")

# Dynamic warning shown when update available:
if newer_version_available:
    st.warning(f"A newer version is available ({latest}). "
               "Run `uv pip install --upgrade pdstools` to update.")
```

❌ **Avoid:**
```python
# Static header with hardcoded command
st.caption(f"pdstools {version} · Keep up to date: `uv pip install --upgrade pdstools`")
```

### Documentation Updates

When updating analysis documentation:
- Keep analysis lists focused on key features
- Match current page structure
- Update descriptions when page names change
- Spell-check all user-facing text
```

**Step 2: Verify the addition**

Run: `grep -A 5 "## Style Guide" CONTRIBUTING.md`
Expected: See the new section with product naming table

**Step 3: Commit CONTRIBUTING.md update**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add style guide section to contributing guide

Adds product naming conventions and documentation standards
for human contributors.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Discover all Decision Analyzer references

**Files:**
- Reference: Multiple files across codebase

**Step 1: Search for user-facing documentation references**

Run: `grep -r "Decision Analyzer" --include="*.rst" --include="*.md" python/docs/`
Expected: Find instances in getting started docs

**Step 2: Search for UI text references**

Run: `grep -rn "Decision Analyzer" --include="*.py" python/pdstools/app/decision_analyzer/`
Expected: Find instances in Home.py and page files

**Step 3: Search for utility text references**

Run: `grep -rn "Decision Analyzer" --include="*.py" python/pdstools/utils/`
Expected: Find any instances in shared utilities

**Step 4: Document findings**

Create a checklist of files to update:
- `python/docs/source/GettingStartedWithDecisionAnalyzer.rst`
- `python/pdstools/app/decision_analyzer/Home.py`
- Any other files found in searches above

Note: DO NOT commit yet - this is discovery only

---

## Task 4: Update GettingStartedWithDecisionAnalyzer.rst

**Files:**
- Modify: `python/docs/source/GettingStartedWithDecisionAnalyzer.rst`

**Step 1: Update body text references**

Find and replace in the file:
- "The Decision Analyzer provides:" → "The Decision Analysis Tool provides:"
- "Launching the Decision Analyzer" → "Launching the Decision Analysis Tool"
- "Using the Decision Analyzer" → "Using the Decision Analysis Tool"

Note: Keep the title as is (already "Decision Analyzer Tool")

**Step 2: Update the analysis types list**

Replace lines 11-20 with updated list emphasizing V2 features:

```rst
The Decision Analysis Tool provides:

- **Overview**: Key metrics and insights about your offer strategy at a glance
- **Action Funnel**: Visualize how offers flow through the full decision pipeline and identify where they drop off
- **Action Distribution**: Understand the distribution of actions at the arbitration stage using interactive treemaps
- **Optionality Analysis**: Analyze the number of actions available per customer and personalization opportunities
- **Global Sensitivity Analysis**: Understand how arbitration factors (propensity, value, levers, context weights) affect decision-making
- **Win/Loss Analysis**: Examine which actions win or lose in arbitration and the factors behind these outcomes
- **Arbitration Component Distribution**: Analyze the distribution of prioritization components to identify potential issues
- **No coding required**: Everything is configured through a web-based UI
- **Interactive visualizations**: Hover over charts for detailed information and insights
```

**Step 3: Verify changes**

Run: `grep "Decision Analyz" python/docs/source/GettingStartedWithDecisionAnalyzer.rst`
Expected: Should only show "Decision Analyzer Tool" (title) and "Decision Analysis Tool" (body text)

**Step 4: Spell-check the modified sections**

Manually review lines 1-252 for typos, especially:
- Lines 11-20 (updated analysis list)
- Any lines with "Decision Analysis Tool"

**Step 5: Commit the documentation update**

```bash
git add python/docs/source/GettingStartedWithDecisionAnalyzer.rst
git commit -m "docs(decision_analyzer): standardize to 'Decision Analysis Tool'

- Update user-facing text to use 'Decision Analysis Tool' consistently
- Revise analysis types list to highlight V2 features (Overview, Action Funnel, Optionality)
- Update page descriptions to match current application structure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update streamlit_utils.py version header

**Files:**
- Modify: `python/pdstools/utils/streamlit_utils.py`

**Step 1: Update the show_version_header function**

Change line 114 from:
```python
st.caption(
    f"pdstools {pdstools_version} · Keep up to date: `uv pip install --upgrade pdstools`",
)
```

To:
```python
st.caption(
    f"pdstools {pdstools_version} · Keep up to date",
)
```

Note: Keep lines 123 and 646 unchanged (they show uv command in dynamic upgrade warnings)

**Step 2: Verify the change**

Run: `grep -n "Keep up to date" python/pdstools/utils/streamlit_utils.py`
Expected: See updated line 114 without the uv command, lines 123 and 646 still have it

**Step 3: Commit the utility update**

```bash
git add python/pdstools/utils/streamlit_utils.py
git commit -m "refactor(streamlit_utils): remove install command from version header

Remove hardcoded 'uv pip install' from static version caption to avoid
assuming user's package manager. Dynamic upgrade warnings still show the
command when appropriate.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Verify no remaining user-facing "Decision Analyzer" references

**Files:**
- Reference: Multiple files for validation

**Step 1: Check documentation files**

Run: `grep -r "Decision Analyzer[^T]" --include="*.rst" --include="*.md" python/docs/`
Expected: No results (or only in contexts where it's correct)

Note: The `[^T]` pattern excludes "Decision Analyzer Tool" which is acceptable

**Step 2: Check Streamlit app files**

Run: `grep -rn '"[^"]*Decision Analyzer[^T]' --include="*.py" python/pdstools/app/decision_analyzer/`
Expected: No results in markdown strings or UI text

**Step 3: Document any remaining issues**

If any user-facing instances found:
- List the files and line numbers
- Determine if they need updating
- Update if necessary following the same commit pattern

**Step 4: No commit needed - validation only**

---

## Task 7: Build and verify documentation

**Files:**
- Build output: `python/docs/build/html/`

**Step 1: Build the Sphinx documentation**

Run: `cd python/docs && make html`
Expected: Build completes with "build succeeded" message, no warnings about GettingStartedWithDecisionAnalyzer.rst

**Step 2: Check for build warnings**

Review output for:
- Broken links
- Formatting issues
- Missing references

Expected: No warnings related to our changes

**Step 3: Visually inspect the rendered HTML**

Run: `open python/docs/build/html/GettingStartedWithDecisionAnalyzer.html`
Or: Check the file in a browser

Verify:
- "Decision Analysis Tool" displays correctly in body text
- Analysis types list is properly formatted
- No rendering issues

**Step 4: No commit needed - validation only**

---

## Task 8: Smoke test the application

**Files:**
- Runtime test: Decision Analysis Tool application

**Step 1: Launch the Decision Analysis Tool**

Run: `pdstools decision_analyzer`
Expected: Application launches and opens in browser

**Step 2: Verify Home page**

Check:
- Page title shows "Decision Analysis"
- Version header shows "pdstools X.X.X · Keep up to date" (NO uv pip install command)
- If upgrade available, warning should show with uv command (dynamic)

**Step 3: Navigate through pages**

Visit each analysis page:
1. Global Data Filters
2. Overview
3. Action Distribution
4. Action Funnel
5. Global Sensitivity
6. Win/Loss Analysis
7. Optionality Analysis
8. Offer Quality Analysis
9. Thresholding Analysis
10. Arbitration Component Distribution
11. About

Verify page titles match the documentation list

**Step 4: Stop the application**

Press Ctrl+C in terminal to stop Streamlit

Expected: Clean shutdown

**Step 5: No commit needed - validation only**

---

## Task 9: Update auto memory with new conventions

**Files:**
- Create/Update: `/Users/perdo/.claude/projects/-Users-perdo-dev-pega-datascientist-tools/memory/MEMORY.md`

**Step 1: Add naming convention to memory**

Add to MEMORY.md:

```markdown
## Product Naming Conventions

- **Decision Analysis Tool**: Always use "Decision Analysis Tool" in user-facing text (docs, UI, help). Use `decision_analyzer` in code/CLI.
- Version headers: Show version + "Keep up to date" message, but NOT hardcoded install commands. Let dynamic checks show upgrade commands.
- See CLAUDE.md and CONTRIBUTING.md style guide sections for full conventions.
```

**Step 2: Verify memory update**

Run: `cat /Users/perdo/.claude/projects/-Users-perdo-dev-pega-datascientist-tools/memory/MEMORY.md | grep -A 3 "Product Naming"`
Expected: See the naming conventions section

**Step 3: Commit memory update**

```bash
git add /Users/perdo/.claude/projects/-Users-perdo-dev-pega-datascientist-tools/memory/MEMORY.md
git commit -m "docs(memory): add Decision Analysis Tool naming convention

Record naming standards in auto memory for future consistency.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Completion Checklist

After completing all tasks, verify:

- [ ] CLAUDE.md created with comprehensive conventions
- [ ] CONTRIBUTING.md updated with style guide section
- [ ] GettingStartedWithDecisionAnalyzer.rst uses "Decision Analysis Tool"
- [ ] Analysis types list matches current pages, highlights V2 features
- [ ] streamlit_utils.py version header simplified
- [ ] Documentation builds without errors
- [ ] Application launches and displays correct naming
- [ ] Auto memory updated with conventions
- [ ] All changes committed with clear messages

## Testing Commands Summary

```bash
# Build documentation
cd python/docs && make html

# Launch application
pdstools decision_analyzer

# Search for remaining references
grep -r "Decision Analyzer[^T]" --include="*.rst" --include="*.md" python/docs/
grep -rn "Decision Analyzer" --include="*.py" python/pdstools/app/decision_analyzer/

# Run tests (if applicable)
pytest python/tests
```

## Notes

- Keep code references as `decision_analyzer` - do not change module names, CLI commands, or file paths
- Only update user-facing text visible to end users
- Style guides are now the source of truth for future consistency
- Auto memory helps maintain conventions across sessions

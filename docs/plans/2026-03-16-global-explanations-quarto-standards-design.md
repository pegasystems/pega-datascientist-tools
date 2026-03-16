# GlobalExplanations Quarto Report Standards Alignment - Design

**Date:** 2026-03-16
**Issue:** [#564](https://github.com/pegasystems/pega-datascientist-tools/issues/564)
**Status:** Design Approved

## Overview

This design document outlines the plan to align GlobalExplanations reports with the established Quarto report standards documented in CLAUDE.md. The goal is to modernize the reports to match the quality and consistency of reference implementations like HealthCheck.qmd and ModelReport.qmd.

### Key Principle

**Minimal, focused changes.** We preserve the existing template generation architecture (string substitution) and enhance only what's necessary to meet standards. No structural refactoring.

## Background

The GlobalExplanations reports currently use a template-based approach where:
1. Template .qmd files contain placeholders (e.g., `{ROOT_DIR}`, `{TOP_N}`)
2. `generate_report.py` reads templates and performs string substitution
3. Generated .qmd files are rendered by Quarto

This approach differs from HealthCheck.qmd which is a direct, self-contained .qmd file. However, the issue scope confirms the template approach should be preserved.

## Current State

### Files in Scope
- `python/pdstools/reports/GlobalExplanations/assets/templates/getting-started.qmd` - Introduction page
- `python/pdstools/reports/GlobalExplanations/assets/templates/overview.qmd` - Model-level analysis
- `python/pdstools/reports/GlobalExplanations/assets/templates/context.qmd` - Single context pages
- `python/pdstools/reports/GlobalExplanations/assets/templates/all_context_header.qmd` - Batch processing header
- `python/pdstools/reports/GlobalExplanations/assets/templates/all_context_content.qmd` - Embedded context code
- `python/pdstools/reports/GlobalExplanations/scripts/generate_report.py` - Generation orchestration

### Missing Standards
- Comprehensive YAML front matter with Pega branding
- Error handling around plot generation (`report_utils.quarto_plot_exception()`)
- Pega template styling for visualizations (`template="pega"`)
- Credits and version information sections
- Guidance callouts for interpretation
- Interactive plot disclaimer
- Consistent formatting and structure

## Design Decisions

### Approach Selection

**Chosen: Direct Template Enhancement (Approach A)**

We considered three approaches:
- **A) Direct Template Enhancement** - Add standards directly into template files
- **B) Template Utilities Layer** - Create abstraction layer for common patterns
- **C) Hybrid Enhancement** - Mix of direct and utility-based

**Rationale for Approach A:**
- Aligns with project standards (transparency, self-contained code)
- Simple to implement and maintain
- No new abstractions needed for 6 templates
- Easy to preview final output
- Minimal scope, avoiding structural changes
- YAGNI principle - we don't need the flexibility of utilities

### Priority Levels

**All 11 improvements will be implemented:**

**High Priority:**
1. Add comprehensive YAML front matter with Pega branding
2. Wrap all plot generation in try/except with `report_utils.quarto_plot_exception()`
3. Add `template="pega"` to all Plotly visualizations
4. Add credits section with `show_credits()` and `show_versions()`

**Medium Priority:**
5. Add guidance callouts for business analyst interpretation
6. Add interactive plot disclaimer callout
7. Improve section structure and markdown formatting
8. Update documentation comments for clarity

**Low Priority:**
9. Ensure consistent title formatting across templates
10. Add proper code cell options (`echo: false`, output control)
11. Standardize spacing and visual hierarchy

### Testing Strategy

**Comprehensive testing at three levels:**
- **Unit tests** - Template reading, string substitution, generated content validation
- **Integration tests** - Full report generation workflow with test parameters
- **End-to-end validation** - Manual verification with sample data

**Coverage target:** 80%+ on template generation logic

## Implementation Plan

### Template-by-Template Changes

#### 1. getting-started.qmd (Introduction Page)

**Changes:**
- Add standard YAML front matter (title-block-banner, theme, CSS)
- Add credits section at end
- Add interactive plot disclaimer callout
- Keep all existing content and placeholders unchanged

**Placeholders preserved:** `{DATE_INFO}`, `{TOP_N}`, `{TOP_K}`, `{CONTRIBUTION_TEXT}`, `{MODEL_CONTEXT_LIMIT}`

#### 2. overview.qmd (Model-Level Overview)

**Changes:**
- Add standard YAML front matter
- Add Python imports block at top with `echo: false`
- Wrap `plot_contributions_for_overall()` calls in try/except
- Add `.update_layout(template="pega")` to figures
- Add credits section at end

**Placeholders preserved:** `{ROOT_DIR}`, `{DATA_FOLDER}`, `{TOP_N}`, `{TOP_K}`, `{CONTRIBUTION_TYPE}`, `{CONTRIBUTION_TEXT}`

#### 3. context.qmd (Single Context Pages)

**Changes:**
- Add minimal YAML front matter (format: html only)
- Keep all embed syntax unchanged
- Keep all content unchanged

**Placeholders preserved:** `{CONTEXT_STR}`, `{EMBED_PATH_FOR_BATCH}`, `{CONTEXT_LABEL}`, `{TOP_N}`, `{CONTRIBUTION_TEXT}`

#### 4. all_context_header.qmd (Batch Processing Header)

**Changes:**
- Wrap plot generation in try/except blocks
- Add `template="pega"` to figures
- Keep all existing logic unchanged

**Placeholders preserved:** `{ROOT_DIR}`, `{DATA_FOLDER}`, `{DATA_PATTERN}`, `{TOP_N}`, `{CONTRIBUTION_TEXT}`

#### 5. all_context_content.qmd (Embedded Context Code)

**Changes:**
- Wrap plot generation in try/except blocks
- Add `template="pega"` to figures
- Keep all existing logic unchanged

**Placeholders preserved:** `{CONTEXT_DICT}`, `{CONTEXT_LABEL}`, `{TOP_N}`, `{TOP_K}`, `{CONTRIBUTION_TYPE}`

#### 6. generate_report.py (Generation Orchestration)

**Changes:**
- Update documentation comments only
- No functional changes to generation logic

### Standard Patterns

#### YAML Front Matter (Main Templates)

```yaml
---
title: "Title Here"
title-block-banner: true
author: "Pega Data Scientist tools"
date: today
format:
  html:
    page-layout: full
    code-fold: true
    standalone: true
    toc: true
    theme:
      light: flatly
    css: assets/pega-report-overrides.css
---
```

#### Error Handling Pattern

```python
try:
    # existing plot code
    fig = explanations.plot.method()
    fig.update_layout(template="pega")
    fig.show()
except Exception as e:
    report_utils.quarto_plot_exception("Plot Name", e)
```

#### Credits Section Pattern

```markdown
# Credits

```{python}
# | echo: false
from pdstools.utils import show_versions
report_utils.show_credits("pega-datascientist-tools/python/pdstools/reports/GlobalExplanations/assets/templates/FILENAME.qmd")
show_versions.show_versions(include_dependencies=False)
```

::: {.callout-note collapse="true"}
## Expand for detailed version information
```{python}
show_versions.show_versions()
```
:::
```

#### Import Block Pattern

```python
```{python}
# | echo: false
# | output: false
from pdstools.utils import report_utils
```
```

#### Interactive Plot Disclaimer

```markdown
::: {.callout-tip}
Charts are built with [Plotly](https://plotly.com/python/) and have [user controls for panning, zooming etc](https://plotly.com/chart-studio-help/zoom-pan-hover-controls/).
These interactive plots do not render well in portals like Sharepoint or Box. View from a browser for best experience.
:::
```

### String Substitution Preservation

All existing placeholders remain unchanged:
- `{ROOT_DIR}`, `{DATA_FOLDER}`, `{TOP_N}`, `{TOP_K}`
- `{CONTRIBUTION_TYPE}`, `{CONTRIBUTION_TEXT}`, `{DATE_INFO}`
- `{CONTEXT_STR}`, `{CONTEXT_LABEL}`, `{EMBED_PATH_FOR_BATCH}`
- `{DATA_PATTERN}`, `{CONTEXT_DICT}`, `{MODEL_CONTEXT_LIMIT}`

New code blocks are inserted around these substitutions without interfering with the generation logic.

## Testing Approach

### Unit Tests
**New file:** `python/tests/reports/test_global_explanations_templates.py`

Test coverage:
- Template files exist and are readable
- Generated .qmd files contain required elements (YAML headers, credits sections, import blocks)
- String substitutions work correctly for all placeholders
- No syntax errors in generated Python code blocks
- Error handling blocks are properly formed

### Integration Tests
**New file:** `python/tests/reports/test_global_explanations_generation.py`

Test coverage:
- `ReportGenerator` instantiation with test parameters
- Full report generation workflow completion
- All expected output files created (getting-started.qmd, overview.qmd, context files, batch files)
- Generated files are valid Quarto documents
- Template substitutions produce correct output

### End-to-End Validation (Manual)
- Run report generation with sample explanations data
- Verify Quarto renders all files without errors
- Visual inspection checklist:
  - Credits sections appear correctly
  - Plots use Pega template styling
  - Error handling catches exceptions gracefully
  - Interactive plot disclaimers visible
  - Version information displays properly
  - YAML front matter renders correctly

### Coverage Target
- Aim for 80%+ coverage on template generation logic
- Focus on critical paths: template reading, string substitution, file writing
- Ensure edge cases are covered (missing params, malformed templates)

## Migration Path

### Phase 1: Template Updates
1. Update getting-started.qmd with YAML and credits
2. Update overview.qmd with error handling and Pega styling
3. Update context.qmd with minimal YAML
4. Update all_context_header.qmd with error handling
5. Update all_context_content.qmd with error handling
6. Update generate_report.py documentation

### Phase 2: Testing
1. Write unit tests for template validation
2. Write integration tests for generation workflow
3. Run manual end-to-end validation

### Phase 3: Documentation
1. Update relevant documentation if needed
2. Add comments to templates explaining standard patterns

## Non-Goals

Explicitly **out of scope** for this work:
- Changing the template generation architecture
- Converting to direct .qmd files (like HealthCheck.qmd)
- Adding new features or analysis sections
- Restructuring the report organization
- Changing the Explanations API or data aggregation logic
- Modifying report rendering parameters

## Success Criteria

The implementation will be considered successful when:
1. All 6 template files include required standards (YAML, error handling, credits, Pega styling)
2. Generated reports render successfully with Quarto
3. Visual inspection confirms alignment with HealthCheck.qmd quality
4. Unit tests achieve 80%+ coverage on generation logic
5. Integration tests verify end-to-end workflow
6. No regressions in existing report functionality

## References

- GitHub Issue: https://github.com/pegasystems/pega-datascientist-tools/issues/564
- CLAUDE.md Quarto Report Standards section
- Reference implementation: python/pdstools/reports/HealthCheck.qmd
- Template directory: python/pdstools/reports/GlobalExplanations/assets/templates/

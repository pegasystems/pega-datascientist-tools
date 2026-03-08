# Decision Analyzer Text Styling Consistency

**Date:** 2026-03-08
**Status:** Approved
**Type:** UI Enhancement

## Problem Statement

The Decision Analyzer pages have inconsistent text styling for explanatory content. Some pages use `st.write()`, others use plain triple-quoted strings, and only some use `st.caption()`. The Action Funnel page in particular has verbose explanatory text that feels overwhelming. The Thresholding Analysis page demonstrates a better pattern with compact `st.caption()` styling that should be applied consistently across all pages.

## Goals

1. **Visual consistency**: All Decision Analyzer pages should use the same text styling patterns
2. **Clear hierarchy**: Page-level introductions should be prominent; section-level explanations should be subdued
3. **Improved compactness**: Reduce visual noise by making explanatory text smaller and more concise
4. **Future-proof**: Document the pattern clearly to prevent future drift

## Design

### Text Styling Hierarchy

**Page-level introduction** (immediately after page title):
- Use triple-quoted strings `"""..."""` for normal body text
- Purpose: High-level page description and key questions
- Should remain prominent to orient users

**Section-level explanations** (inside `st.container(border=True)` blocks):
- Use `st.caption()` for smaller, subdued text
- Purpose: Describe specific charts, tables, or interactions
- Should be concise (30-50 words) and visually de-emphasized

### Conversion Patterns

**Pattern 1: Triple-quoted strings inside containers**

Before:
```python
with st.container(border=True):
    "## Section Title"
    """
    Explanation text here.
    Can span multiple lines.
    """
```

After:
```python
with st.container(border=True):
    "## Section Title"

    st.caption(
        "Explanation text here. "
        "Can span multiple lines."
    )
```

**Pattern 2: st.write() with triple-quoted strings**

Before:
```python
st.write("""
Track how many offers **enter** each stage...
""")
```

After:
```python
st.caption(
    "Track how many offers **enter** each stage..."
)
```

**Pattern 3: Action Funnel condensing example**

Before (3 paragraphs, ~80 words):
```python
st.write("""
Track how many offers **enter** each stage of your decisioning pipeline. The numbers show
offers arriving at each stage (not exiting) — the narrowing funnel reveals where offers drop
off, helping you spot bottlenecks or overly aggressive filtering.

The funnel height shows **Average Actions per Interaction** (how many times offers appear per customer
interaction on average), while **Reach** indicates what percentage of interactions had at least one
offer of that type. Together, these metrics reveal both frequency and penetration of your offers.

Use **Granularity** (sidebar) to analyze at different levels — from high-level offer categories
(Issue/Group) down to individual actions or treatments.
""")
```

After (condensed, ~45 words):
```python
st.caption(
    "Track offers entering each stage. Funnel height shows **Average Actions per Interaction**, "
    "while **Reach** shows percentage of interactions with at least one offer. Use **Granularity** "
    "(sidebar) to analyze from high-level categories down to individual actions."
)
```

### Key Principles

- Preserve markdown formatting (bold, italics) within captions
- Remove redundant explanations
- Keep critical information (metric definitions, interaction tips)
- Use `st.caption()` syntax with string continuation for readability

## Implementation Approach

### Step 1: Identify all files requiring updates

Use grep to find:
- Pages with `st.write("""...)` patterns
- Pages with triple-quoted strings inside `st.container(border=True)` blocks
- Pages with section explanations that aren't using `st.caption()`

Expected files: ~10-12 pages in `python/pdstools/app/decision_analyzer/pages/*.py` plus `Home.py`

### Step 2: Systematic page-by-page review

For each page:
1. Read the entire file
2. Identify page-level intro (keep as `"""..."""`)
3. Identify section-level explanations inside containers
4. Convert section explanations to `st.caption()`
5. Condense verbose explanations where needed
6. Preserve all markdown formatting

### Step 3: Update CLAUDE.md

Add the "Streamlit Text Styling" subsection to the existing "Decision Analyzer Standards" section with:
- Explanation of the hierarchy
- Code examples
- Writing guidelines

### Step 4: Testing

Manual smoke test:
- Launch `pdstools decision_analyzer` with sample data
- Navigate through all pages
- Verify text renders correctly with smaller font
- Ensure markdown formatting (bold, italics) still works
- Check visual hierarchy (page intros prominent, section text subdued)

## Documentation Update

Add to CLAUDE.md under "Decision Analyzer Standards":

```markdown
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

    st.plotly_chart(fig, width="stretch")
```

**Writing Guidelines:**
- Keep captions concise (aim for 30-50 words per section)
- Preserve markdown formatting (bold, italics) within captions
- Explain what the visualization shows and how to use it
- Remove redundant information that's obvious from context
```

## Risk Assessment

**Low risk:**
- Non-functional changes only
- No logic changes
- No data processing changes
- Pure UI styling improvements
- `st.caption()` is a well-established Streamlit API

**Testing strategy:**
- Manual visual verification across all pages
- No automated tests needed (visual change only)

## Benefits

1. **Professional appearance**: Smaller, subdued text for explanations creates a more polished UI
2. **Better readability**: Clear visual hierarchy helps users distinguish page purpose from section details
3. **Reduced clutter**: Condensing verbose text makes pages easier to scan
4. **Consistency**: All pages follow the same pattern established by Thresholding Analysis
5. **Maintainability**: Clear documentation prevents future inconsistencies

## Non-Goals

- No changes to page functionality
- No changes to data processing
- No new features or capabilities
- No changes to existing charts or visualizations

## References

- Reference implementation: `python/pdstools/app/decision_analyzer/pages/10_Thresholding_Analysis.py`
- Primary update target: `python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py`

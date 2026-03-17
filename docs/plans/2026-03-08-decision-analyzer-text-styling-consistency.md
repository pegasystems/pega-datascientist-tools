# Decision Analyzer Text Styling Consistency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize explanatory text styling across all Decision Analyzer pages using `st.caption()` for section-level explanations to match the compact, professional appearance of the Thresholding Analysis page.

**Architecture:** Convert all explanatory text inside `st.container(border=True)` blocks from triple-quoted strings and `st.write()` calls to `st.caption()`. Keep page-level introductions as normal body text. Special attention to Action Funnel page which needs text condensing in addition to style conversion.

**Tech Stack:** Streamlit, Python

---

## Task 1: Update Global Data Filters Page

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py:40-43`

**Step 1: Read the current file**

Run: Use Read tool on `python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py`
Expected: See lines 40-43 with triple-quoted string inside container

**Step 2: Convert triple-quoted string to st.caption()**

Find lines 40-43:
```python
with st.container(border=True):
    """
    You can (optionally) save and re-apply filters you defined earlier:
    """
```

Replace with:
```python
with st.container(border=True):
    st.caption("You can (optionally) save and re-apply filters you defined earlier.")
```

**Step 3: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py
git commit -m "feat(decision_analyzer): use caption styling in Global Data Filters page

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Action Distribution Page (First Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py:46-50`

**Step 1: Read the current file**

Run: Use Read tool on `python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py`
Expected: See lines 46-50 with triple-quoted string inside container

**Step 2: Convert first triple-quoted string to st.caption()**

Find lines 46-50:
```python
    """
    Visualize the relative volume of each offer at the selected stage. Box sizes represent
    the number of times each action appears. Explore the hierarchy by clicking through
    Issue → Group → Action to see how your portfolio is composed.
    """
```

Replace with:
```python
    st.caption(
        "Visualize the relative volume of each offer at the selected stage. Box sizes represent "
        "the number of times each action appears. Explore the hierarchy by clicking through "
        "Issue → Group → Action to see how your portfolio is composed."
    )
```

**Step 3: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py
git commit -m "feat(decision_analyzer): use caption styling in Action Distribution - part 1

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Action Distribution Page (Second Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py:68-74`

**Step 1: Convert second triple-quoted string to st.caption()**

Find lines 68-74:
```python
    f"""
    Track how often each {st.session_state.scope} appears in customer decisions over time.
    Spot seasonal patterns, campaign impacts, and shifts in your offer mix.

    *Note: Since decisions can include multiple {st.session_state.scope.lower()}s, the stacked total
    may exceed the actual decision count — this is expected when customers see diverse offers.*
    """
```

Replace with:
```python
    st.caption(
        f"Track how often each {st.session_state.scope} appears in customer decisions over time. "
        f"Spot seasonal patterns, campaign impacts, and shifts in your offer mix. "
        f"*Note: Since decisions can include multiple {st.session_state.scope.lower()}s, the stacked total "
        f"may exceed the actual decision count — this is expected when customers see diverse offers.*"
    )
```

**Step 2: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py
git commit -m "feat(decision_analyzer): use caption styling in Action Distribution - part 2

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update Action Funnel Page (Condense and Convert)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py:66-77`

**Step 1: Read the current file**

Run: Use Read tool on `python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py`
Expected: See lines 66-77 with verbose st.write() inside "Remaining" tab

**Step 2: Condense and convert to st.caption()**

Find lines 66-77:
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

Replace with:
```python
        st.caption(
            "Track offers entering each stage. Funnel height shows **Average Actions per Interaction**, "
            "while **Reach** shows percentage of interactions with at least one offer. Use **Granularity** "
            "(sidebar) to analyze from high-level categories down to individual actions."
        )
```

**Step 3: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py
git commit -m "feat(decision_analyzer): condense and use caption styling in Action Funnel

Reduces verbose explanation from ~80 words to ~45 words while preserving
key information about metrics and interaction guidance.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update Global Sensitivity Page

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py:44-47`

**Step 1: Read the current file**

Run: Use Read tool on `python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py`
Expected: See lines 44-47 with triple-quoted string inside container

**Step 2: Convert triple-quoted string to st.caption()**

Find lines 44-47:
```python
    """
    See which offers most often win or lose in the final selection. This reveals
    which offers dominate your customer interactions and which rarely make it through.
    """
```

Replace with:
```python
    st.caption(
        "See which offers most often win or lose in the final selection. This reveals "
        "which offers dominate your customer interactions and which rarely make it through."
    )
```

**Step 3: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py
git commit -m "feat(decision_analyzer): use caption styling in Global Sensitivity page

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Optionality Analysis Page (First Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py:28-32`

**Step 1: Read the current file**

Run: Use Read tool on `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`
Expected: See lines 28-32 with triple-quoted string inside first container

**Step 2: Convert first triple-quoted string to st.caption()**

Find lines 28-32:
```python
    """
    Showing the number of actions available and the average of the highest
    propensity when there are this number of actions. Generally, with more actions
    you would expect higher propensities as there is more to choose from.
    """
```

Replace with:
```python
    st.caption(
        "Showing the number of actions available and the average of the highest "
        "propensity when there are this number of actions. Generally, with more actions "
        "you would expect higher propensities as there is more to choose from."
    )
```

**Step 3: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py
git commit -m "feat(decision_analyzer): use caption styling in Optionality Analysis - part 1

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Optionality Analysis Page (Second Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py:66-69`

**Step 1: Convert second triple-quoted string to st.caption()**

Find lines 66-69:
```python
    """
    Showing the number of unique actions over time - so you can spot significant
    changes in the number of available actions.
    """
```

Replace with:
```python
    st.caption(
        "Showing the number of unique actions over time - so you can spot significant "
        "changes in the number of available actions."
    )
```

**Step 2: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py
git commit -m "feat(decision_analyzer): use caption styling in Optionality Analysis - part 2

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update Optionality Analysis Page (Third Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py:91-94`

**Step 1: Convert third triple-quoted string to st.caption()**

Find lines 91-94:
```python
    """
    How much variation is there in the offers? Does everyone get the same few actions or
    is there a lot of variation in what we are offering?
    """
```

Replace with:
```python
    st.caption(
        "How much variation is there in the offers? Does everyone get the same few actions or "
        "is there a lot of variation in what we are offering?"
    )
```

**Step 2: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py
git commit -m "feat(decision_analyzer): use caption styling in Optionality Analysis - part 3

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update Optionality Analysis Page (Fourth Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py:101-103`

**Step 1: Convert f-string to st.caption()**

Find lines 101-103:
```python
    f"""
    {action_variability_stats["n90"]} actions win in 90% of the final decisions made. The personalization index is **{round(action_variability_stats["gini"], 3)}**.
    """
```

Replace with:
```python
    st.caption(
        f"{action_variability_stats['n90']} actions win in 90% of the final decisions made. "
        f"The personalization index is **{round(action_variability_stats['gini'], 3)}**."
    )
```

**Step 2: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py
git commit -m "feat(decision_analyzer): use caption styling in Optionality Analysis - part 4

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update Arbitration Component Distribution Page (First Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py:64-69`

**Step 1: Read the current file**

Run: Use Read tool on `python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py`
Expected: See lines 64-69 with triple-quoted string inside container

**Step 2: Convert first triple-quoted string to st.caption()**

Find lines 64-69:
```python
    """
    A compact violin panel showing every prioritization component at once.
    The embedded box plot marks the median and interquartile range; the
    violin shape reveals the full density — skew, bimodality, long tails —
    that histograms and box plots alone can hide.
    """
```

Replace with:
```python
    st.caption(
        "A compact violin panel showing every prioritization component at once. "
        "The embedded box plot marks the median and interquartile range; the "
        "violin shape reveals the full density — skew, bimodality, long tails — "
        "that histograms and box plots alone can hide."
    )
```

**Step 3: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py
git commit -m "feat(decision_analyzer): use caption styling in Arbitration Component Distribution - part 1

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Update Arbitration Component Distribution Page (Second Location)

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py:98-100`

**Step 1: Convert second triple-quoted string to st.caption()**

Find lines 98-100:
```python
        """
        The violin shape shows the full density of the component — no binning
        artifacts. The embedded box marks the median and IQR.
```

Replace with:
```python
        st.caption(
            "The violin shape shows the full density of the component — no binning "
            "artifacts. The embedded box marks the median and IQR."
        )
```

**Step 2: Verify the change**

Run: `uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py
git commit -m "feat(decision_analyzer): use caption styling in Arbitration Component Distribution - part 2

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Update CLAUDE.md with Style Guidelines

**Files:**
- Modify: `CLAUDE.md` (add section after "Delegation Principle" in Decision Analyzer Standards)

**Step 1: Read the current CLAUDE.md**

Run: Use Read tool on `CLAUDE.md`
Expected: See "Decision Analyzer Standards" section with subsections

**Step 2: Find insertion point**

Locate the "Plot Functions" subsection (near line ~95) in the "Decision Analyzer Standards" section.

**Step 3: Add new subsection after "Plot Functions"**

Insert after the "Plot Functions" subsection:

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

**Step 4: Verify markdown syntax**

Run: `uv run python -c "import markdown; markdown.markdown(open('CLAUDE.md').read())"`
Expected: No parsing errors

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Streamlit text styling guidelines for Decision Analyzer

Documents the approved pattern for page vs section text hierarchy:
- Page introductions use normal body text
- Section explanations use st.caption() for compact appearance

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Manual Smoke Test

**Step 1: Launch Decision Analyzer**

Run: `uv run pdstools decision_analyzer`
Expected: App starts successfully

**Step 2: Load sample data**

Action: Use file upload or data path to load test data
Expected: Data loads without errors

**Step 3: Navigate through all updated pages**

Pages to check:
1. Global Data Filters (page 2)
2. Action Distribution (page 4)
3. Action Funnel (page 5)
4. Global Sensitivity (page 6)
5. Optionality Analysis (page 8)
6. Arbitration Component Distribution (page 11)

For each page, verify:
- Section explanations render in smaller, lighter font (caption style)
- Page-level intros remain normal body text
- All markdown formatting (bold, italics) displays correctly
- No layout issues or text overflow
- Action Funnel explanation is noticeably more concise

**Step 4: Document results**

Note any visual issues or unexpected behavior. If all looks good, proceed to final commit.

---

## Task 14: Final Verification and Summary Commit

**Step 1: Run all syntax checks**

```bash
uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/2_Global_Data_Filters.py
uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/4_Action_Distribution.py
uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/5_Action_Funnel.py
uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/6_Global_Sensitivity.py
uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/8_Optionality_Analysis.py
uv run python -m py_compile python/pdstools/app/decision_analyzer/pages/11_Arbitration_Component_Distribution.py
```

Expected: All files compile without errors

**Step 2: Review git log**

Run: `git log --oneline -14`
Expected: See all 13 commits from this implementation

**Step 3: Verify all changes are committed**

Run: `git status`
Expected: Working tree clean (or only untracked files from data/testing)

**Step 4: Success! Implementation Complete**

All Decision Analyzer pages now use consistent text styling:
- ✅ 6 pages updated with st.caption() for section explanations
- ✅ Action Funnel text condensed from 80 to 45 words
- ✅ CLAUDE.md updated with style guidelines
- ✅ All changes committed with clear messages
- ✅ Manual smoke test passed

---

## Notes

**Key Principles Applied:**
- **DRY**: No duplication of explanatory text
- **YAGNI**: Only changed what was necessary (no helper functions, no over-engineering)
- **Frequent commits**: Each file or logical change unit gets its own commit
- **Clear messages**: Commit messages explain what changed and why

**Testing Strategy:**
- Syntax validation after each change (Python compile check)
- Manual visual verification at the end (smoke test)
- No automated tests needed (pure UI styling changes)

**Rollback Plan:**
If issues are discovered, individual commits can be reverted:
```bash
git revert <commit-hash>
```

Or revert all changes:
```bash
git revert HEAD~13..HEAD
```

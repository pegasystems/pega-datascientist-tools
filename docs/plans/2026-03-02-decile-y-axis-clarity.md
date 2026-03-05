# Decile Y-Axis Clarity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve clarity of decile overview plots in Thresholding Analysis by updating labels and adding explanatory caption.

**Architecture:** Simple label updates in the plotting method and adding a Streamlit caption to explain the dual-axis visualization.

**Tech Stack:** Python, Plotly, Streamlit

---

## Task 1: Update Plot Labels in plots.py

**Files:**
- Modify: `python/pdstools/decision_analyzer/plots.py:16-42`
- Test: Manual verification (plot testing would require complex mocking of DecisionAnalyzer data)

**Step 1: Read the current implementation**

Read the file to understand the current structure:
```bash
# Already done in exploration phase
```

**Step 2: Update the bar trace name**

In `python/pdstools/decision_analyzer/plots.py` line 22, change:
```python
# FROM:
fig.add_trace(go.Bar(x=df["Decile"], y=df["Count"], name="Impressions"))

# TO:
fig.add_trace(go.Bar(x=df["Decile"], y=df["Count"], name="Actions Below Threshold"))
```

**Step 3: Update the y-axis title**

In `python/pdstools/decision_analyzer/plots.py` line 36, change:
```python
# FROM:
yaxis_title="Volume",

# TO:
yaxis_title="Action Count",
```

**Step 4: Verify the changes**

Read the file to confirm both changes are correct:
```bash
grep -A 2 "fig.add_trace(go.Bar" python/pdstools/decision_analyzer/plots.py
grep "yaxis_title" python/pdstools/decision_analyzer/plots.py
```

Expected: See "Actions Below Threshold" and "Action Count"

**Step 5: Commit the label changes**

```bash
git add python/pdstools/decision_analyzer/plots.py
git commit -m "feat(decision_analyzer): clarify decile plot y-axis labels

Replace vague 'Volume'/'Impressions' labels with clear 'Action Count'/
'Actions Below Threshold' to match terminology used throughout the UI.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Explanatory Caption to Thresholding Analysis

**Files:**
- Modify: `python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py:217-230`

**Step 1: Add caption after the section header**

In `python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py`, after line 218:
```python
with st.container(border=True):
    "## Decile Overviews"

    # ADD THIS:
    st.caption(
        "Each bar shows the count of actions with values below that decile threshold. "
        "The line shows the threshold value at each decile."
    )

    dec_col1, dec_col2 = st.columns(2)
```

Full context (lines 217-230 after change):
```python
# ---------------------------------------------------------------------------
# Section 4: Decile charts (static context)
# ---------------------------------------------------------------------------
with st.container(border=True):
    "## Decile Overviews"

    st.caption(
        "Each bar shows the count of actions with values below that decile threshold. "
        "The line shows the threshold value at each decile."
    )

    dec_col1, dec_col2 = st.columns(2)
    with dec_col1:
        st.plotly_chart(
            da.plot.threshold_deciles("Propensity", "Propensity"),
            use_container_width=True,
        )
    with dec_col2:
        st.plotly_chart(
            da.plot.threshold_deciles("Priority", "Priority"),
            use_container_width=True,
        )
```

**Step 2: Verify the changes**

Read the file around line 218 to confirm the caption was added correctly:
```bash
sed -n '217,230p' python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py
```

Expected: See the caption between the header and column creation

**Step 3: Commit the caption addition**

```bash
git add python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py
git commit -m "feat(decision_analyzer): add explanatory caption to decile plots

Add caption explaining the dual-axis decile overview visualization to help
users understand what the bars and line represent.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Manual Verification

**Step 1: Run the Decision Analyzer app**

```bash
cd python/pdstools/app/decision_analyzer
streamlit run Home.py
```

**Step 2: Navigate to Thresholding Analysis**

1. Load sample data if needed
2. Navigate to "Thresholding Analysis" page
3. Scroll to "Decile Overviews" section

**Step 3: Verify the changes**

Check that:
- [ ] Left y-axis says "Action Count" (not "Volume")
- [ ] Legend shows "Actions Below Threshold" (not "Impressions")
- [ ] Caption appears between header and plots
- [ ] Caption text matches: "Each bar shows the count of actions with values below that decile threshold. The line shows the threshold value at each decile."

**Step 4: Stop the app**

Press Ctrl+C in the terminal

---

## Summary

**Changes Made:**
1. Updated plot y-axis label from "Volume" to "Action Count"
2. Updated bar trace name from "Impressions" to "Actions Below Threshold"
3. Added explanatory caption to help users interpret the dual-axis visualization

**Files Modified:**
- `python/pdstools/decision_analyzer/plots.py` (2 lines)
- `python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py` (5 lines added)

**Testing:**
- Manual verification via running the Streamlit app

**Commits:** 2 commits following conventional commit format

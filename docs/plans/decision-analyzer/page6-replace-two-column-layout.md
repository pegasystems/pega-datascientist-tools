# Replace 2-column layout with side-by-side bars (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`, `plots.py`

The current Streamlit two-column layout for "wins to" vs "loses from" is difficult to compare. The dual-bar layout used in Global Sensitivity is more readable.

## Approach

Replace the two-column view with a single dual-bar chart (grouped or diverging bars) that directly compares wins and losses side-by-side.

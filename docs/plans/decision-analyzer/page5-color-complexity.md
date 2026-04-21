# Limit color complexity at Action level (Page 5 — Global Sensitivity)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/5_Global_Sensitivity.py`, `plots.py`

Action-level coloring becomes unreadable when there are many actions (distinct colours become indistinguishable).

## Approach

Cap at top-N actions (e.g. top 10) with a distinct colour each; fall back to a single neutral colour for all others. Add a UI note when the cap is applied.

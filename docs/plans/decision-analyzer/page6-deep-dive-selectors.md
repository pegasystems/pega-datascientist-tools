# Deep-dive selectors for specific opponents (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`

No way to scope the deep-dive to specific counterpart items the comparison group wins against or loses to.

## Approach

Add selectors for "wins to" and "loses from" counterparts. Scope the deep-dive analysis to the selected items.

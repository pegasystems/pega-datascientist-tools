# Verify numbers across visualizations (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`

Bar charts and box plots on the Win/Loss page sometimes show different counts for the same data slice. Root cause unknown.

## Approach

Audit data sources for each chart; ensure all views are computed from the same filtered dataset.

# Consistent colors across both plots (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`, `plots.py`

The two charts on the Win/Loss page currently choose colours independently, so the same action may appear in different colours in each chart.

## Approach

Compute a shared categorical colour mapping at the page level and pass it into both chart functions.

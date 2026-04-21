# Audit "rank of winning" usage (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`, `decision_analyzer.py`

"Rank of winning" may not be correctly plumbed through all win/loss analyses.

## Approach

Trace the "rank of winning" field from data ingestion through each win/loss method and verify it appears correctly in all visualizations.

# Use aggregated data instead of fresh sampling (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`

Page 6 currently re-samples on each render rather than reusing `DecisionAnalyzer`'s aggregated/sampled data.

## Approach

Reuse the aggregated/sampled data already held in the `DecisionAnalyzer` instance. Consistent with how other pages access data.

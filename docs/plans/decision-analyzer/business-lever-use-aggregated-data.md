# Business Lever: use aggregated/sampled data

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/_stashed/Business_Lever_Analysis.py`

The stashed page bypasses the normal data flow and re-processes the raw dataset.

## Approach

Reuse `DecisionAnalyzer`'s aggregated/sampled data, consistent with all other pages.

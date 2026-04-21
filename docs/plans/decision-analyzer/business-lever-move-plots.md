# Business Lever: move plots to `plots` module

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/_stashed/Business_Lever_Analysis.py`, `plots.py`

Inline chart definitions make the page hard to test and reuse.

## Approach

Extract to `plots.py` as reusable functions before re-enabling the page.

# Verify many-channel handling (Page 6 — Win/Loss Analysis)

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`, `plots.py`

Faceting (e.g. `pyChannel/pyDirection`) has not been validated with high-cardinality channel sets.

## Approach

Test with a dataset that has 5+ channels/directions. Check whether facet labels overflow, whether performance degrades, and whether the layout remains readable.

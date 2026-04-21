# Stratified sampling

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/decision_analyzer.py`

Current sampling is random. For analysis across channels/directions, random sampling may under-represent small groups.

## Approach

Add optional `sample_stratified_by` parameter for proportional representation across channels/directions. Use polars group-aware sampling.

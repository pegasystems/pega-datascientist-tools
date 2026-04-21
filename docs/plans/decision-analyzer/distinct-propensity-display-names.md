# Distinct propensity display names

**Priority:** P2
**Touches:** `python/pdstools/decision_analyzer/column_schema.py`

`column_schema.py` maps both model propensity and final propensity to the same display name, making them indistinguishable in the UI.

## Approach

Use separate display names: `"Model Propensity"` vs `"Final Propensity"`.

# Handle no-action / single-action edge cases (Page 8 — Offer Quality Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/8_Offer_Quality_Analysis.py`, `DecisionAnalyzer.py`

The offer quality analysis breaks or produces misleading output when an interaction has no actions, only one action, or only low-propensity options.

## Approach

Generalise the analysis to handle these edge cases gracefully. Add guard clauses and informational messages for degenerate inputs.

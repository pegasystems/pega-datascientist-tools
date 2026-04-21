# Handle mandatory actions

**Priority:** P1
**Touches:** `python/pdstools/decision_analyzer/decision_analyzer.py`, Streamlit pages

Priority 4999999+ bypasses normal ranking, skewing sensitivity and win/loss analysis. `DecisionAnalyzer` already accepts `mandatory_expr` and sorts by `is_mandatory`.

## Approach

- Auto-detect mandatory actions from priority value (≥ 4999999).
- Flag them in UI visualizations (e.g. distinct marker/colour).
- Exclude or separate in sensitivity analysis so they don't distort rank distributions.

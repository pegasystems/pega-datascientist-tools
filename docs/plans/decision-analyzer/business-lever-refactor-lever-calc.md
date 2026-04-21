# Business Lever: refactor lever calculation into DecisionAnalyzer

**Priority:** P1
**Touches:** `python/pdstools/app/decision_analyzer/_stashed/Business_Lever_Analysis.py`, `decision_analyzer.py`

The lever calculation logic is entirely embedded in the stashed page code, making it untestable.

## Approach

Extract into `DecisionAnalyzer` as a method (e.g. `get_lever_analysis`). Add unit tests. Page becomes a thin rendering layer.

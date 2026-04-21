# Dynamic stage UI

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/`, `plots.py`

Hardcoded stage names remain in the Optionality page (`stage="Output"`) and `plots.py` (`"Final"` filter).

## Approach

Replace with data-driven `DecisionAnalyzer.stages` throughout. Audit all pages and plot functions for hardcoded stage references.

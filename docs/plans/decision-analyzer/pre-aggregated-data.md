# Pre-aggregated data support

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`

Infinity Action Analysis can produce pre-computed results. Currently `DecisionAnalyzer` always re-aggregates.

## Approach

Detect pre-aggregated format at load time and bypass the aggregation step, using the pre-computed results directly.

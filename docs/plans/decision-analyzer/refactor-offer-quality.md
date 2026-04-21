# Refactor `get_offer_quality`

**Priority:** P2
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`

`get_offer_quality` uses a manual stage loop instead of delegating to `aggregate_remaining_per_stage`, leading to duplicated aggregation logic.

## Approach

Rewrite to use the shared aggregation helper; verify results match existing behaviour; update tests.

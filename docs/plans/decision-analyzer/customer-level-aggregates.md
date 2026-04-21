# Customer-level aggregates

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/decision_analyzer.py`

Per-customer statistics are not yet implemented. Postponed until a representative multi-customer dataset is available for testing and validation.

## Approach

Once a suitable dataset exists, add aggregation methods that group by customer ID and expose per-customer metrics (e.g. actions seen, win rates, propensity distributions).

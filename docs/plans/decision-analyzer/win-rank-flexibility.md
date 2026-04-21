# Win rank flexibility

**Priority:** P2
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`

`get_win_loss_distribution_data` has a hardcoded `max_value=10` for rank cap.

## Approach

Compute the max rank from actual data; return all ranks and let the UI filter/cap as needed. This avoids silently dropping data when users have deeper rank distributions.

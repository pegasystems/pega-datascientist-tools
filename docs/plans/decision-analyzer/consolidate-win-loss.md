# Consolidate win/loss methods

**Priority:** P2
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`

`_winning_from` / `_losing_to` duplicate logic — only the rank comparison differs. Similarly `get_win_loss_distribution_data` and `get_win_loss_distributions` have overlapping concerns.

## Approach

- Merge `_winning_from` / `_losing_to` into a single private helper with a `direction` param.
- Review whether `get_win_loss_distribution_data` and `get_win_loss_distributions` can be unified into one function with a flag or post-processing step.

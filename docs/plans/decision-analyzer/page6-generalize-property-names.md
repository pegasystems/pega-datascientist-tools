# Generalize and relabel arbitration property names (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`

Arbitration property labels are repeated and may match mock-data naming rather than real Pega field names.

## Approach

Audit property labels; derive them from `DecisionAnalyzer`'s column schema rather than hardcoding strings.

# Infer top-X and win rank bounds from data (Page 5 — Global Sensitivity)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/5_Global_Sensitivity.py`

top-X and win rank bounds are currently user-specified or hardcoded. Users don't know appropriate values for their data.

## Approach

Auto-compute from actual max rank present in the dataset. Pre-populate the UI controls with data-driven defaults. Matches the core library item `win-rank-flexibility`.

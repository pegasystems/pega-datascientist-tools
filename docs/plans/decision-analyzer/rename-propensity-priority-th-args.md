# Rename `propensityTH` / `priorityTH` to snake_case

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`,
`python/pdstools/decision_analyzer/plots.py`, Streamlit pages 2 & 8, tests.

`DecisionAnalyzer.filtered_action_counts` (and the matching `plots.py`
helpers) take `propensityTH` and `priorityTH`. These violate
PEP 8 / our snake_case naming convention. They were left in place during
the v5 gold-standard pass to keep the diff focused — renaming forces
plots.py changes (held back for a separate split agent) plus updates to
Streamlit pages and ~15 test call sites.

## Approach

Rename to `propensity_threshold` / `priority_threshold` once the
plots.py split lands. Update all callers in the same PR.

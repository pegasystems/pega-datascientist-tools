# Pie chart stage limit

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/plots.py`

The stage pie chart has a hardcoded cap at 5 stages, silently dropping additional stages.

## Approach

Make the cap dynamic based on data. For >5 stages, switch to a bar chart (more readable for many categories). Add a note when stages are collapsed.

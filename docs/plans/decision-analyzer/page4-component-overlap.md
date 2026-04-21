# Component overlap analysis (Page 4 — Action Funnel)

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/pages/4_Action_Funnel.py`, `plots.py`

No current visualisation shows which actions are hit by multiple filters simultaneously.

## Approach

Add an UpSet plot or Sankey diagram showing filter co-occurrence — which actions are removed by multiple components.

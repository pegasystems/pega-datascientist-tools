# Make colors stay legible at low counts (Page 6 — Win/Loss Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/6_Win_Loss_Analysis.py`, `plots.py`

Colours fade to near-invisible when bin counts are small (opacity scales with count).

## Approach

Clamp minimum opacity / use outlines so bars remain visually distinct even at count = 1.

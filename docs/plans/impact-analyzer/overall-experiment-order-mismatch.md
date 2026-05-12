# Overall Summary — experiment ordering mismatch between forest plot and cards

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py`

The forest plot silently drops experiments whose `lift` or `se` are null
(e.g. when there is no data for that experiment). The experiment cards and
the bottom summary table keep all rows. Users see the forest plot listing
N rows in one order and the cards below listing M ≥ N rows in the same
order, which reads as a reorder.

## Approach

- Render every experiment on the forest plot, using a placeholder marker
  ("no data") for null-metric rows so the plot and card grid stay aligned.
- Or: drop the same null-metric experiments from the card grid and the
  detailed table, and surface a single banner listing the dropped
  experiments and the reason.
- Pick whichever approach is least surprising to a user comparing the
  forest plot row count to the card count.

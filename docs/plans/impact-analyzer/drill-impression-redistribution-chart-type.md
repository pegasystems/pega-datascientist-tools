# Drill Down — review Impression Redistribution chart type

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/pages/2_Drill_Down.py`

The current "Impression Redistribution" section uses a grouped horizontal
bar chart comparing Test vs Control impression share per channel. Two
concerns:

1. The third arm (NBA) is not represented as its own group — depending on
   which experiment is selected, the "Test" group *is* NBA, but the
   labelling doesn't make that clear.
2. Grouped bars are an awkward way to convey "where did NBA redirect
   impressions vs where did the control variant send them" — a Sankey
   (control → channels vs NBA → channels) or a divergent-bar chart
   (gain/loss per channel) would convey the redistribution story more
   directly.

## Approach

This is open to the implementer. Investigate which chart type best
conveys the redistribution story for the available data:

- Sankey (control vs NBA → channels).
- Divergent bar chart (per-channel delta in impression share).
- Stacked bar chart (each channel sums to 100 %, split by arm).
- Keep grouped bars but make the NBA-vs-control labelling explicit and
  add a third bar for any experiment that has three arms.

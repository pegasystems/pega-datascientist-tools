# Drill Down — review utility of two detail tables

**Priority:** P3
**Touches:** `python/pdstools/app/impact_analyzer/pages/2_Drill_Down.py`

The Drill Down page renders two channel-level detail views:

1. **Channel Performance Details** — per-channel expanders with metrics
   row, arm comparison table, and step-by-step formula walkthroughs.
2. **Detailed Per-Channel Metrics** — a flat dataframe table with the
   same numbers in a single grid.

Both convey similar information. Users have flagged that they aren't
sure how useful either is in their current form.

## Approach

- Confirm with users which view they actually open during analysis (per
  channel or the flat table).
- Consolidate into one (e.g. keep the flat table for export / scanning,
  drop the per-channel expanders or move them behind a single "Show
  formula walkthrough" toggle).
- Or: keep both but make the section collapsed by default and clearly
  labelled so the page focuses on the overview / cannibalization
  sections above.

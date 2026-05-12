# Drill Down — add per-channel trend lines

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/pages/2_Drill_Down.py`,
`python/pdstools/impactanalyzer/ImpactAnalyzer.py`

The Drill Down page is purely cross-sectional — every chart shows the
current state of each channel. The Overall Summary page has trend
charts (lift over time) but Drill Down has none.

## Approach

- Add a per-channel lift trend chart, faceted by channel, similar to the
  existing Overall Summary trend chart but using
  `summarize_experiments(by=["SnapshotTime", "Channel"])`.
- Place it near the channel performance details so users can correlate
  current state with the trajectory.
- Apply EWMA smoothing consistent with Overall Summary.

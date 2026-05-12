# Channels — add per-channel trend lines

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/pages/2_Channels.py`,
`python/pdstools/impactanalyzer/ImpactAnalyzer.py`

The Channels page is purely cross-sectional — every chart shows the
current state of each channel. The Overall Summary page has trend
charts (lift over time) but the Channels page has none.

## Approach

- Add a per-channel lift trend chart, faceted by channel, similar to the
  existing Overall Summary trend chart but using
  `summarize_experiments(by=["SnapshotTime", "Channel"])`.
- Apply EWMA smoothing consistent with Overall Summary.

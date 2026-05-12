# Overall Summary — per-channel trend in experiment cards

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py`,
`python/pdstools/impactanalyzer/Plots.py`

The channel selector at the top of the lift chart now drives the chart,
the experiment cards, and the summary table (filters all of them to the
selected channel). Trend sparklines inside each card are still
suppressed when a channel is selected, because `_trend_df` is computed
as `summarize_experiments(by="SnapshotTime")` — aggregated across
channels — and showing it next to per-channel card metrics would be
misleading.

## Approach

- Compute a per-channel trend frame:
  `ia.summarize_experiments(by=["Channel", "SnapshotTime"])`.
- Filter to the selected channel and pass it as `trend_df` to
  `_render_experiment_card` instead of `None`.
- Update `_chart_lift_trend` if needed so it picks the right rows for
  `(experiment, channel)` combinations.
- Consider also faceting the Detailed Metrics table by the selected
  channel (currently it always renders all channels broken out).


# Overall Summary — surface Channel in experiment cards and tables

**Priority:** P1
**Touches:** `python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py`,
`python/pdstools/impactanalyzer/Plots.py`

The Overall Summary page currently aggregates every experiment across all
channels in the forest plot, the experiment cards, and the trend charts.
Channel only appears in the bottom "Detailed Metrics" table. A short-term
selector above the forest plot was added (filters the plot only) — see the
selector wired through `summarize_experiments(by="Channel")` in
`pages/1_Overall_Summary.py`.

For multi-channel data, channel is a primary dimension. A user looking at
"NBA vs Random" with five channels needs to see the per-channel breakdown
without scrolling to a separate table.

## Approach

- Make the channel selector at the top of the page apply consistently to:
  - Forest plot (already done as short-term fix).
  - Experiment cards (`_render_experiment_card` in
    `pages/1_Overall_Summary.py`).
  - Trend charts inside each card (`_build_trend_chart`).
  - The "Detailed Metrics" table at the bottom.
- Alternatively, render small-multiples / faceted versions of the cards and
  trend charts when "All channels" is selected (Plotly `facet_col=Channel`).
- Decide how to handle experiments where one channel has no data (drop vs.
  show empty card with explanatory text — see related plan
  `overall-experiment-order-mismatch.md`).

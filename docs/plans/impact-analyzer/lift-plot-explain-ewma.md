# Trend charts — explain the EWMA smoothed line

**Priority:** P3
**Touches:** `python/pdstools/app/impact_analyzer/pages/1_Overall_Summary.py`

Each experiment card renders a lift trend with two lines: raw daily lift
and an EWMA (exponentially weighted moving average). Non-statistical
users have no idea what the green smoothed line represents — the legend
just says "Smoothed" and the formula reference is in a separate expander
at the bottom of the page.

## Approach

- Add a one-sentence caption directly under the trend chart explaining
  EWMA in plain English ("smoothed line down-weights older days") with a
  link or anchor to the formula reference.
- Or: rename the trace to something less mathematical
  ("Trend (EWMA-smoothed)") and rely on hover for the math.
- Apply the same treatment to the value-lift trend chart.

# Hover info in stacked histograms

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/plots.py`

Hovering over a stacked histogram bar only shows the segment value, not the total bar height.

## Approach

Add a custom hover template that shows both the segment value and the total bar height. Use Plotly's `customdata` / `hovertemplate` mechanism.

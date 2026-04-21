# `over_time` multi-by support

**Priority:** P3
**Touches:** `python/pdstools/adm/Plots.py`

`ADMDatamart.plot.over_time` only accepts a single `by` column. `HealthCheck.qmd` and `ModelReport.qmd` would benefit from grouping by multiple dimensions (e.g. `["Channel", "Direction"]`).

## Approach

Generalise the `by` parameter to accept a list or polars expression, mirroring the facet pattern used in the bubble chart. Ensure backward compatibility when a single string is passed.

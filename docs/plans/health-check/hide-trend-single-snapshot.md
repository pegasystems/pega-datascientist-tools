# Hide trend plots for single-snapshot data

**Priority:** P2
**Touches:** `reports/HealthCheck.qmd`, `reports/ModelReport.qmd`

Trend plots render as empty charts or single isolated dots when only one snapshot is present in the data. This is confusing and looks like a bug.

## Approach

Detect `n_unique(SnapshotTime) == 1` at report-render time. Either hide the trend sections entirely or replace them with an informational callout (e.g. "Trend plots require data from multiple snapshots").

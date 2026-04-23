# `adm/Analysis.py` — audit overlap with `adm/Aggregates.py`

Priority: P3

Files touched: `python/pdstools/adm/Analysis.py`,
`python/pdstools/adm/Aggregates.py`.

## Status of the initial audit

Done as part of the PR #623 architecture-review pass. **Verdict: minimal
overlap; no immediate refactor warranted.** Recording findings here in
case future work uncovers a reason to reconsider.

## Per-method check

| `Analysis._check_*` | Aggregation it does | Already delegates? |
|---|---|---|
| `_get_last_data` | `dm.aggregates.last()` + null/nan fill | ✅ thin wrapper |
| `_check_channels` | `dm.aggregates.summary_by_channel(query=...)` | ✅ delegates |
| `_check_configurations` | `dm.unique_configurations` iteration | N/A — config inspection, not aggregation |
| `_check_model_maturity` | operates on `last_data` (already from aggregates) | ✅ via `_get_last_data` |
| `_check_model_performance` | single weighted avg via `cdh_utils.weighted_average_polars` | not aggregate-table material |
| `_check_taxonomy` | `report_utils.n_unique_values` + `combined_data` filter | distinct util, not duplication |
| `_check_response_distribution` | Gini on `last_data["ResponseCount"]` / `["Positives"]` | unique computation |
| `_check_trends` | per-snapshot `group_by(["SnapshotTime", "Channel/Direction"])` weighted avg | **borderline**, see below |
| `_check_predictors` | `dm.aggregates.predictors_global_overview()` + extra `combined_data` filter | ✅ delegates |
| `_check_predictions` | operates on `prediction.summary_by_channel` | N/A — prediction object, not datamart |

## The one borderline case: `_check_trends`

Computes `weighted_average_polars("Performance", "ResponseCount")` per
`(SnapshotTime, Channel/Direction)` cell. `Aggregates.summary_by_channel`
does the same per `(Channel, Direction)` *across* snapshots, and
`Aggregates.last()` collapses to one snapshot. There is no existing
"per-snapshot per-channel time series" aggregate.

If `Aggregates.over_time(by=[...])` (a sibling of the existing
`plot.over_time`) ever lands as a public aggregate, `_check_trends`
should switch to delegate. Until then, the current inline group-by is
the path of least resistance.

## Action

No code changes recommended. Re-evaluate when:
- A new check needs the per-snapshot-per-channel trend data and would
  benefit from a shared cached aggregate.
- `Aggregates` grows an `over_time` / `trend` method for other reasons
  (e.g. Streamlit usage).

## Cross-refs

- AGENTS.md "Extend before you create"
- `python/pdstools/adm/Aggregates.py` — current public aggregate surface

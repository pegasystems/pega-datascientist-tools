# Consistent active model filtering

**Priority:** P2
**Touches:** `python/pdstools/adm/Plots/ (package)`, `reports/HealthCheck.qmd`, `reports/ModelReport.qmd`

Related: [#594](https://github.com/pegasystems/pega-datascientist-tools/issues/594)

`active_models_filter_expr` is applied inconsistently across plots — some charts filter it, others don't, and it's not obvious which without reading the source. The `threshold_updated_days` parameter is also not prominently communicated.

## Approach

- Audit every chart's data scope; annotate each with whether `active_models_filter_expr` is applied.
- Make `threshold_updated_days` more prominent in the report (e.g. a callout box at the top of the report).
- Standardise filter application; document the intended policy in the ADM module.

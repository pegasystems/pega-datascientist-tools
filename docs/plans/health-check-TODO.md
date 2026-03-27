# ADM Health Check — Backlog

Open work items for the ADM Health Check report and supporting ADM library code.

**Priority levels:** P1 = high / P2 = medium / P3 = nice-to-have

---

## Plots & Plotly Template

- [ ] **[P2] Improve "pega" Plotly template** ([#600](https://github.com/pegasystems/pega-datascientist-tools/issues/600)) — Template only sets colorway and hovermode. 120+ ad-hoc `update_layout()` calls across `adm/Plots.py`, `HealthCheck.qmd`, `ModelReport.qmd`. Add sensible defaults for legend positioning, axis automargin, font sizes, facet label styling. Create `apply_pega_report_layout(fig)` utility.

- [ ] **[P2] Consistent legend coloring across partitioned plots** — Same predictor category gets different colors in different partitions (`_boxplot_pre_aggregated` uses local `fixed_colors` with index-shifting fallback). Reuse `pdstools/utils/color_mapping.py` (`create_categorical_color_mappings()`) — already proven in Decision Analyzer's `color_mappings` property. Compute global mapping at `ADMDatamart` level; pass `color_discrete_map` into box plot functions.

- [ ] **[P3] Standardize long axis label abbreviation** — Duplicated `s[:25] + "..."` truncation in `BinAggregator.py` and `Plots.py`. Extract to shared `abbreviate_labels()` utility; apply to predictor names, action names, config names.

---

## Quarto Reports

- [ ] **[P2] Hide trend plots for single-snapshot data** — Trend plots render as empty or single dots when only one snapshot. Detect `n_unique(SnapshotTime) == 1` and either hide sections or show informational callout.

- [ ] **[P2] Consistent active model filtering** ([#594](https://github.com/pegasystems/pega-datascientist-tools/issues/594)) — `active_models_filter_expr` applied inconsistently across plots. Audit and annotate each chart's data scope; make `threshold_updated_days` more prominent.

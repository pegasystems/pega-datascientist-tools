# ADM Health Check (Quarto Report) — Backlog

Active work items for the ADM Health Check report (`HealthCheck.qmd` / `ModelReport.qmd`) and supporting ADM library code.

## Quick Reference

**Blockers (fix before release):**
- None currently

**High Impact (fix soon):**
- None currently

**Priority levels:**
- **[P1]** High — Critical functionality, poor UX, or blocking issues
- **[P2]** Medium — Important improvements, moderate UX impact
- **[P3]** Low — Nice-to-have, minor issues, future enhancements

**Total items:** 5

---

## Plots (`adm/Plots.py`) and Plotly Template (`utils/pega_template.py`)

### Medium Priority

- [ ] **[P2] Improve "pega" Plotly template to reduce ad-hoc layout patching** ([GitHub issue #600](https://github.com/pegasystems/pega-datascientist-tools/issues/600)) — The custom `pega` Plotly template (`python/pdstools/utils/pega_template.py`) is minimal: it only sets a colorway and `hovermode`. It does not configure legend positioning, margin/padding, axis automargin, title truncation, font sizes, or facet label behavior. As a result, nearly every plot in the codebase requires ad-hoc `update_layout()`, `update_xaxes()`, `update_yaxes()`, `update_traces()` calls to prevent legends being cut off, titles overlapping axes, labels overflowing containers, and faceted plots sizing incorrectly. Across `adm/Plots.py`, `HealthCheck.qmd`, and `ModelReport.qmd` alone there are **120+ such patch calls**, plus helpers like `fig_update_facet()` that exist solely to work around template gaps. **Consequence:** Charts frequently render with clipped legends, overlapping labels, or misfit sizing — especially in the Quarto HTML reports where the viewport width is fixed; each new plot requires duplicating the same workarounds; visual inconsistency across the codebase; maintenance burden. **Scope:** This affects all pdstools tools (Health Check, Model Report, Decision Analyzer plots, Global Explanations), not just the Health Check. **Action:** (1) Audit the most common `update_layout`/`update_xaxes`/`update_yaxes` patterns across the codebase and identify which can be absorbed into the template. (2) Enhance the `pega` template with sensible defaults for: legend position and overflow (`legend=dict(orientation="h", yanchor="bottom", ...)`), axis automargin (`xaxis=dict(automargin=True), yaxis=dict(automargin=True)`), consistent font sizes, title truncation behavior, and facet label styling. (3) Create a utility function (e.g., `apply_pega_report_layout(fig, ...)`) in `pdstools/utils/` for common report-specific adjustments (fixed width for Quarto HTML, responsive height scaling) so that per-plot patching is a single call rather than repeated inline code. (4) Gradually migrate existing plots to use the improved template, removing redundant `update_*` calls. **Files involved:** `python/pdstools/utils/pega_template.py` (template definition), `python/pdstools/adm/Plots.py`, `python/pdstools/reports/HealthCheck.qmd`, `python/pdstools/reports/ModelReport.qmd`, `python/pdstools/decision_analyzer/plots.py`.

- [ ] **[P2] Consistent legend coloring across partitioned plots** — The same legend item (e.g. a predictor category like "IH" or "Customer") can have different colors in different plots within the same report. This is most visible in the predictor performance box plots generated via `partitioned_plot` → `predictor_performance`, where each partition independently assigns colors in `_boxplot_pre_aggregated` using a local `fixed_colors` dict and a `template_colors` fallback list. Because the set of categories varies per partition, the fallback index shifts and the same category ends up with different colors across plots. **Consequence:** Confusing for readers scanning across predictor box plots in the Health Check; harder to visually track a predictor category across configurations/channels; looks unpolished. **Existing solution to reuse:** `pdstools/utils/color_mapping.py` provides `create_categorical_color_mappings()`, a generic utility built for Decision Analyzer's `color_mappings` property . It computes a single deterministic color mapping from *all* unique values in the dataset (sorted alphabetically, modulo-indexed into a colorway). This already lives in `pdstools/utils/` and is app-agnostic. **Action:** (1) Compute a global color mapping for `PredictorCategory` (and any other legend dimension like `Configuration`, `Channel`) once, at report or `ADMDatamart` level, using `create_categorical_color_mappings()` over the full dataset. (2) Pass the resulting `color_discrete_map` into `_boxplot_pre_aggregated` instead of the current ad-hoc `fixed_colors`/`template_colors` logic. (3) Remove the TODO-marked inline color code from `_boxplot_pre_aggregated`. (4) Optionally expose a `color_mappings` cached property on `ADMDatamart` similar to `DecisionAnalyzer.color_mappings` so the Quarto reports and the Streamlit Health Check app both benefit. **Files involved:** `python/pdstools/adm/Plots.py` (`_boxplot_pre_aggregated`, `predictor_performance`, `partitioned_plot`), `python/pdstools/adm/ADMDatamart.py` (optional cached property), `python/pdstools/reports/HealthCheck.qmd` (verify result). **References:** `python/pdstools/utils/color_mapping.py`, `python/pdstools/decision_analyzer/DecisionAnalyzer.py` (`color_mappings` property), `DecisionAnalyzer.color_mappings` implementation.

### Low Priority

- [ ] **[P3] Standardize long axis label abbreviation** — Predictor names and other axis labels (e.g. bin symbols, action names, configuration names) can be very long, causing overlapping or clipped text in plots. There is existing abbreviation code — `BinSymbolAbbreviated` truncation at 25 chars — but it is **duplicated** in both `BinAggregator.py` and `Plots.py` (hardcoded `s[:25] + "..."` / `pl.col("BinSymbol").str.len_chars() < 25`), and it only covers bin symbols, not predictor names or other long labels. Other plots (predictor box plots, heatmaps, bar charts) have no abbreviation at all and rely on Plotly automargin or manual sizing, which often fails. **Consequence:** Inconsistent label handling; some plots clip, some overlap, some abbreviate; duplicated abbreviation logic is fragile; no way for users to control truncation length. **Scope:** Affects Health Check, Model Report, and potentially Decision Analyzer plots. **Action:** (1) Extract a reusable `abbreviate_labels(col, max_len=25)` utility into `pdstools/utils/` that handles truncation with "…" suffix and ensures uniqueness (append index if truncated values collide). (2) Replace duplicated inline truncation in `BinAggregator.py` and `Plots.py` with the shared utility. (3) Apply consistently to predictor names, action names, configuration names, and any other axis labels that can exceed ~30 characters. (4) Consider making `max_len` configurable per plot or as a report parameter. **Files involved:** `python/pdstools/adm/BinAggregator.py`, `python/pdstools/adm/Plots.py`, new utility in `python/pdstools/utils/`.

---

## Quarto Reports (`reports/HealthCheck.qmd`, `reports/ModelReport.qmd`)

### Medium Priority

- [ ] **[P2] Hide or warn about trend plots when data has a single snapshot** — The Health Check report contains several trend/over-time plots: action trend (weekly new/used actions), success rate over time (`over_time` metric="SuccessRate"), model performance over time (`over_time` metric="Performance"), number of empty/immature models over time, response counts over time, and (when Prediction data is available) lift trend and response count trend. When the input data contains only a single snapshot, these plots render as empty or as a single meaningless dot. **Consequence:** Confusing for readers; empty charts with axes but no data look broken; clutters the report with useless sections; no explanation of why the plots are empty. **Action:** (1) Detect early in the report whether the data spans multiple snapshots (e.g. `datamart.model_data.select(pl.col("SnapshotTime").n_unique()).collect().item() > 1`). (2) For single-snapshot data, either hide the trend sections entirely (using Quarto conditional content blocks) or replace them with an informational callout (e.g. `quarto_callout_info("Trend analyses require multiple snapshots over time. Only a single snapshot was found in this dataset.")`). (3) Consider grouping all trend plots under a single "Trends Over Time" section that can be conditionally shown/hidden as a unit. **Files involved:** `python/pdstools/reports/HealthCheck.qmd` (trend plot sections), possibly `python/pdstools/adm/Plots.py` (`over_time` method could return `None` for single-snapshot data).

- [ ] **[P2] Consistent and transparent active model filtering** — The report uses an `active_models_filter_expr` based on `LastUpdate > max(LastUpdate) - threshold_updated_days` (default 30 days) to approximate which models are actively used. However, this filter is applied inconsistently: some plots use it (bubble chart, channel summary, model performance facets) while others operate on the full dataset. Furthermore, the report does not clearly communicate *per plot* what data scope applies — readers cannot tell whether a specific chart shows all models or only recently-updated ones, nor what the effective date range is. **Consequence:** Misleading interpretation; readers may compare numbers across plots without realizing they cover different model populations; the "active" heuristic itself is a workaround (ADM snapshots lack true active/inactive status) and needs to be called out more prominently. **Action:** (1) Audit every plot and table in `HealthCheck.qmd` for whether `active_models_filter_expr` is applied and document the intended scope. (2) Apply the filter consistently — or, where showing all models is intentional, annotate clearly. (3) Add a visible subtitle or annotation to each plot/table indicating the data scope, e.g. "Models with response updates in {date_from} – {date_to}" or "All models (including inactive)". (4) Consider making the threshold configurable as a report parameter (it already is via `threshold_updated_days` but this should be more prominent). **References:** [GitHub issue #594](https://github.com/pegasystems/pega-datascientist-tools/issues/594). **Files involved:** `python/pdstools/reports/HealthCheck.qmd`.

---

## Streamlit App (`app/health_check/`)

*(No items currently)*

---

## Documentation

*(No items currently)*

---

## Testing

*(No items currently)*

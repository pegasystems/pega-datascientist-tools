# ADM Health Check — Backlog

Open work items for the ADM Health Check report and supporting ADM library code.

**Priority levels:** P1 = high / P2 = medium / P3 = nice-to-have

---

## Plots & Plotly Template

- [ ] **[P2] Improve "pega" Plotly template** ([#600](https://github.com/pegasystems/pega-datascientist-tools/issues/600)) — Template only sets colorway and hovermode. 120+ ad-hoc `update_layout()` calls across `adm/Plots.py`, `HealthCheck.qmd`, `ModelReport.qmd`. Add sensible defaults for legend positioning, axis automargin, font sizes, facet label styling. Create `apply_pega_report_layout(fig)` utility.

- [ ] **[P2] Consistent legend coloring across partitioned plots** — Same predictor category gets different colors in different partitions (`_boxplot_pre_aggregated` uses local `fixed_colors` with index-shifting fallback). Reuse `pdstools/utils/color_mapping.py` (`create_categorical_color_mappings()`) — already proven in Decision Analyzer's `color_mappings` property. Compute global mapping at `ADMDatamart` level; pass `color_discrete_map` into box plot functions.

- [x] **[P3] Standardize long axis label abbreviation** — `abbreviate_label` /
  `abbreviate_label_expr` in `utils/plot_utils.py` are the canonical helpers.
  HealthCheck.qmd now uses `abbreviate_label(..., from_end=True)` instead of
  inline slicing. Remaining work (ensuring truncated labels stay unique within
  a plot) is captured by the inline TODO at `BinAggregator.py` line ~822.

---

## Quarto Reports

- [ ] **[P2] Hide trend plots for single-snapshot data** — Trend plots render as empty or single dots when only one snapshot. Detect `n_unique(SnapshotTime) == 1` and either hide sections or show informational callout.

- [ ] **[P2] Consistent active model filtering** ([#594](https://github.com/pegasystems/pega-datascientist-tools/issues/594)) — `active_models_filter_expr` applied inconsistently across plots. Audit and annotate each chart's data scope; make `threshold_updated_days` more prominent.

- [ ] **[P2] Inline CSS in CDN mode reports** — With `embed-resources: false` (CDN mode), Quarto leaves CSS as external `<link>` tags that break when the HTML is copied out of the temp directory. Result: plots render fine (Plotly loads from CDN) but text styling is lost (no Bootstrap theme, no pega overrides).

  **Root cause:** `run_quarto` in `report_utils.py` renders into `temp_dir`. Quarto creates a support directory (e.g. `HealthCheck_files/libs/`) containing CSS files. The HTML references them via relative `<link>` tags. When `Reports.py` copies only the HTML to the output directory, these references break.

  **Five CSS files affected (from actual CDN-mode output):**
  1. `HealthCheck_files/libs/bootstrap/bootstrap-*.min.css` — flatly theme
  2. `HealthCheck_files/libs/bootstrap/bootstrap-icons.css` — icon font
  3. `HealthCheck_files/libs/quarto-html/quarto-syntax-highlighting-*.css`
  4. `HealthCheck_files/libs/quarto-html/tippy.css` — tooltip styling
  5. `assets/pega-report-overrides.css` — Pega logo, typography, colors

  **Fix:** Add `_inline_css(html_path, base_dir)` in `run_quarto` after Quarto completes (when `full_embed=False` and `output_type="html"`). Use a regex to find `<link rel="stylesheet" href="...">` tags, resolve relative paths against `temp_dir`, read the CSS, and replace each `<link>` with a `<style>` block. Skip absolute URLs. Total CSS is ~100KB — negligible vs. the ~7MB HTML.

  **Why not just set `embed-resources: true`?** That triggers Quarto's esbuild pipeline to bundle JavaScript. DJS Docker images removed esbuild due to CVE issues (see #620).

- [ ] **[P3] Expose `full_embed` as CLI option and UI advanced setting** — The `full_embed` flag (default `False`) controls whether JavaScript libraries are bundled into the HTML or loaded from CDN. Consider adding `--full-embed` to the CLI and an advanced toggle in the Streamlit Reports page so users in air-gapped environments can opt into standalone output. Currently the Streamlit app hardcodes `full_embed=True`.

- [ ] **[P3] `over_time` multi-by support** — `ADMDatamart.plot.over_time` only accepts a single `by` column. HealthCheck.qmd and ModelReport.qmd would benefit from grouping by multiple dimensions (e.g. `["Channel", "Direction"]`). Generalize to accept a list or polars expression, mirroring the facet pattern used in the bubble chart.

- [ ] **[P3] Move report-local plot helpers into `adm/Plots.py`** — `ModelReport.qmd` defines several plotly figures inline (cumulative gains/lift area charts, base-propensity hline overlay on predictor binning, the "Philip Mann" lift bar plot duplicated between BinAggregator.py and the qmd). Lift these into reusable functions on `ADMDatamart.plot` so they can be tested and shared.

- [x] **[P3] Replace silent ValueError swallow in `predictors_overview`** —
  Narrowed: now only catches the specific `ValueError` from `last()` when no
  predictor data is loaded, logs it at debug level, and returns `None`.

- [x] **[P3] Auto-detect `name` ending in `.html`** — `get_output_filename`
  now returns `name` verbatim when it already ends with the configured
  output extension (case-insensitive), so `name="report.html"` no longer
  produces `report.html.html`.

# Improve "pega" Plotly template

**Priority:** P2
**Touches:** `python/pdstools/adm/Plots/ (package)`, `reports/HealthCheck.qmd`, `reports/ModelReport.qmd`

Related: [#600](https://github.com/pegasystems/pega-datascientist-tools/issues/600)

The `"pega"` Plotly template only sets `colorway` and `hovermode`. There are 120+ ad-hoc `update_layout()` calls scattered across `adm/Plots/ (package)`, `HealthCheck.qmd`, and `ModelReport.qmd` that each try to normalise legend positioning, axis automargin, font sizes, and facet label styling independently.

## Approach

- Add sensible defaults for legend positioning, axis automargin, font sizes, and facet label styling to the `"pega"` template definition.
- Create an `apply_pega_report_layout(fig)` utility function for any remaining per-figure tweaks that can't be expressed in the template.
- Gradually replace the 120+ ad-hoc `update_layout()` calls with the utility function.

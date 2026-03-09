# Impact Analyzer — Backlog

Active work items for the Impact Analyzer tool.

## Quick Reference

**Blockers (fix before release):**
- None currently

**High Impact (fix soon):**
- None currently

**Priority levels:**
- **[P1]** High — Critical functionality, poor UX, or blocking issues
- **[P2]** Medium — Important improvements, moderate UX impact
- **[P3]** Low — Nice-to-have, minor issues, future enhancements

**Total items:** 2 across core library and app architecture

---

## Core Library

### Medium Priority

- [ ] **[P2] Excel file support** — Currently only supports JSON, CSV, Parquet, Feather, IPC, and ZIP formats. **Consequence:** Users with Excel exports from PDC (e.g., `ImpactAnalyzerExport_*.xlsx`) cannot load data directly; must manually convert to supported format first; friction in workflow for users who receive Excel reports. **Investigation findings:** (1) Polars can read Excel via `pl.read_excel()` with `fastexcel` dependency, (2) Excel exports have different structure than JSON/VBD formats (wide format with aggregated Test/Control columns, multiple sheets with different data), (3) Main data appears in Sheet 4 with columns like `ExperimentName`, `AggregatedAcceptRate_Test`, `AggregatedAcceptRate_Control`, etc., (4) Sheet 7 contains time-series data with Date column, (5) Excel format lacks key columns expected by `ImpactAnalyzer` (no `ControlGroup`, `Channel`, `SnapshotTime` columns; no separate impression counts). **Open questions:** (1) Is Excel export a summary report or does it contain granular control-group-level data? (2) Does Sheet 7 contain the detailed data needed for analysis? (3) Can we map Excel structure to internal data model, or is it fundamentally incompatible? **Action:** (1) Add `fastexcel>=0.19.0` to dependencies in `pyproject.toml`, (2) Analyze all sheets in sample Excel file to understand complete data structure, (3) Determine if Excel format is compatible with internal data model or if it's a different export type, (4) If compatible: extend `read_ds_export()` in `pega_io/File.py` to handle `.xlsx`/`.xls`/`.xlsm` extensions, (5) If compatible: create `from_excel()` classmethod or `_normalize_excel_ia_data()` transformation logic to convert Excel structure to expected format, (6) If incompatible: document limitation and provide conversion guidance. **References:** Investigation session 2026-03-09; sample file `/Users/perdo/Library/CloudStorage/OneDrive-PegasystemsInc/AI Chapter/projects/Impact Analyzer/data/Data maxis/ImpactAnalyzerExport_30days_20240722T054808 - Copy.xlsx`.

### Low Priority

- [ ] **[P3] Interaction History support** — `from_ih()` method is not yet implemented. **Consequence:** Users with raw Interaction History data cannot use Impact Analyzer; must use VBD or PDC exports only; limits adoption for users who want to analyze historical experiments from IH. **Action:** Implement `from_ih()` method to load and transform IH data into Impact Analyzer format; determine which IH columns map to required fields (`ControlGroup`, `Impressions`, `Accepts`, `ValuePerImpression`); add tests with sample IH data; document any limitations or required preprocessing.

---

## Streamlit App — Architecture

*(No items currently)*

---

## Streamlit Pages

*(No items currently)*

---

## Plots (`Plots.py`)

*(No items currently)*

---

## Documentation

*(No items currently)*

---

## Testing

*(No items currently)*

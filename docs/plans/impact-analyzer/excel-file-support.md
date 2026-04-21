# Excel file support

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/`, `python/pdstools/impact_analyzer/`

Users with PDC Excel exports (`ImpactAnalyzerExport_*.xlsx`) cannot load data directly. Excel exports have a different structure: wide format, aggregated Test/Control columns, and multiple sheets.

## Approach

- Investigate whether the Excel format is compatible with the internal data model or is fundamentally a different export type.
- If compatible, extend `read_ds_export()` with `fastexcel` as an optional dependency.
- If incompatible, document clearly which export formats are supported and how to convert Excel to a supported format.

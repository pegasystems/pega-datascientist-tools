# Impact Analyzer — Backlog

Open work items for the Impact Analyzer tool.

**Priority levels:** P1 = high / P2 = medium / P3 = nice-to-have

---

## Core Library

- [ ] **[P2] Excel file support** — Users with PDC Excel exports (`ImpactAnalyzerExport_*.xlsx`) cannot load data directly. Excel has different structure (wide format, aggregated Test/Control columns, multiple sheets). Needs investigation: determine if Excel format is compatible with internal data model or is fundamentally a different export type. If compatible, extend `read_ds_export()` with `fastexcel` dependency.

- [ ] **[P3] Interaction History support** — `from_ih()` not yet implemented. Determine which IH columns map to required fields; add tests with sample data.

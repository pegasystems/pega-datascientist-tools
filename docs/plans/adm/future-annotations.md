# Add `from __future__ import annotations` across `adm/`

**Priority:** P3
**Touches:** every `.py` under `python/pdstools/adm/`

Currently only `ADMDatamart.py` has it. Adding to `Aggregates.py`,
`Plots.py`, `BinAggregator.py`, `Reports.py`, `ADMTrees.py` lets us
drop string-quoted forward refs and is a one-liner per file.

## Approach

For each file that lacks the import, add it as the very first import
line (after the module docstring if present). Skip `__init__.py` files
that have no non-import code.

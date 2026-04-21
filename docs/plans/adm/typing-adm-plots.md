# `Plots.py` pre-existing mypy errors

**Priority:** P2
**Touches:** `python/pdstools/adm/Plots.py`

Pre-existing mypy errors at lines 751, 1144, 1573, 1596, 1604
(arg-type / assignment).

## Approach

Triage as part of the planned `typing-adm-plots` PR (separate from
the `typing-aggregates` PR that already landed cleanly).

Apply the same patterns used in `typing-aggregates`: prefer fixing
signatures over `# type: ignore`; use the `_require_*` helpers on
`ADMDatamart` for `Optional` narrowing; use `cast()` where genuine
narrowing is needed.

# ADM module — TODO and improvements

Living backlog for the `python/pdstools/adm/` module. This module is the
reference implementation for the namespace-facade pattern (see
AGENTS.md "Namespace facade for large analyzer classes"). Items here
either bring the module further into compliance with its own
conventions or DRY-up duplicated patterns.

Priority levels: P1 = high, P2 = medium, P3 = nice-to-have.

## Open items

### Conventions / consistency

- [ ] **P2 — Lazy plotly imports in `Plots.py`.** Plotly is currently
  imported at module level inside a `try/except ImportError`
  (`adm/Plots.py:35-40`). This violates the AGENTS.md "lazy imports
  inside the method" rule for optional deps, and it means importing
  `ADMDatamart` pulls plotly even when no plot is rendered. Move the
  imports inside each plot method, or have `Plots` properly leverage
  the existing `LazyNamespace` base class for deferred resolution.
- [ ] **P3 — Rename `BinAggregator(dm=...)` → `BinAggregator(datamart=...)`.**
  Every other sub-namespace in `ADMDatamart` takes a `datamart=`
  argument; `BinAggregator` is the lone outlier (`ADMDatamart.py:149`).
  Trivial rename, removes a needless inconsistency.
- [ ] **P3 — Add `from __future__ import annotations` to all `adm/`
  modules.** Currently only `ADMDatamart.py` has it. Adding it to
  `Aggregates.py`, `Plots.py`, `BinAggregator.py`, `Reports.py`,
  `ADMTrees.py` lets us drop string-quoted forward refs and is a
  one-liner per file.
- [ ] **P3 — Add Examples sections to `Plots` method docstrings.** The
  `ADMDatamart` classmethods are exemplary (numpydoc with runnable
  Examples); most `Plots` methods skip the Examples block. Filling
  these in is high-value for users browsing API docs.

### DRY-ups

- [ ] **P3 — Factor out Performance auto-rescale.** The
  `50–100 → 0.5–1.0` rescale block is duplicated verbatim in
  `_validate_model_data` and `_validate_predictor_data`
  (`ADMDatamart.py:474-479` and `:512-518`). Pull into a single
  `_normalize_performance_scale(df)` helper.

### Pre-existing typing debt (out of scope until DA sweep is merged)

- [ ] **P2 — `Plots.py` pre-existing mypy errors** at lines 751, 1144,
  1573, 1596, 1604 (arg-type / assignment). Triage as part of the
  planned `typing-adm-plots` PR.

## Done

- [x] **P2 — Extract `_require_*` helpers** — added
  `_require_model_data`, `_require_predictor_data`, and
  `_require_first_action_dates` on `ADMDatamart`. Eliminates
  `# type: ignore[union-attr]` across `Aggregates.py` and gives users
  an actionable `ValueError` instead of an `AttributeError` on `None`.
- [x] **P3 — DRY-up the `unique_*` cached properties** — collapsed all
  five (`unique_channels`, `unique_configurations`, etc.) onto a
  single `_unique_sorted_from_model_data(expr, alias)` helper.
- [x] **P2 — Resolved `ADMDatamart.py:737, 759` mypy errors** — gone
  with the `_require_*` helpers above.

---

## Cross-references

- AGENTS.md → "Namespace facade for large analyzer classes"
- AGENTS.md → "I/O lives in classmethods, not `__init__`"
- AGENTS.md → "`return_df` parameter on plot methods"
- `docs/plans/decision-analyzer-TODO.md` — DA module backlog (DA is
  in the process of adopting these same conventions; see the
  decoupling-from-Streamlit issue and the typing sweep PR series)

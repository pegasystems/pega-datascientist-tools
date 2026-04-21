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

- [ ] **P2 — Extract `_require_model_data()` / `_require_predictor_data()`
  helpers.** Multiple sites in `Plots.py` and `Aggregates.py` carry
  `# type: ignore[union-attr]` because `model_data` is typed
  `pl.LazyFrame | None`. A small helper that returns the LazyFrame
  or raises a clear `ValueError("Model data not available — …")`
  eliminates the type:ignores and replaces silent `None`-propagation
  with an actionable error.
- [ ] **P3 — Factor out Performance auto-rescale.** The
  `50–100 → 0.5–1.0` rescale block is duplicated verbatim in
  `_validate_model_data` and `_validate_predictor_data`
  (`ADMDatamart.py:474-479` and `:512-518`). Pull into a single
  `_normalize_performance_scale(df)` helper.
- [ ] **P3 — DRY-up the `unique_*` cached properties.** The five
  `cached_property` lookups (`unique_channels`,
  `unique_configurations`, `unique_channel_direction`,
  `unique_configuration_channel_direction`,
  `unique_predictor_categories`) are all the same shape:
  `.select(pl.col(X).unique().sort()).collect()[X].to_list()`. A
  generic `_unique_sorted(col_or_expr)` helper compresses all five
  into one-liners while keeping the cached_properties as the public
  surface.

### Pre-existing typing debt (out of scope until DA sweep is merged)

- [ ] **P2 — `ADMDatamart.py:737, 759` mypy errors.** "Item 'None' of
  'LazyFrame | None' has no attribute 'select'". Resolved
  automatically once `_require_model_data()` helper above is in
  place.
- [ ] **P2 — `Plots.py` pre-existing mypy errors** at lines 751, 1144,
  1573, 1596, 1604 (arg-type / assignment). Triage as part of the
  planned `typing-adm-plots` PR.

## Done

(none yet — file initialized 2026-04-21)

---

## Cross-references

- AGENTS.md → "Namespace facade for large analyzer classes"
- AGENTS.md → "I/O lives in classmethods, not `__init__`"
- AGENTS.md → "`return_df` parameter on plot methods"
- `docs/plans/decision-analyzer-TODO.md` — DA module backlog (DA is
  in the process of adopting these same conventions; see the
  decoupling-from-Streamlit issue and the typing sweep PR series)

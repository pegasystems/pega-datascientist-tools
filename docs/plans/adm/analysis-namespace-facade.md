# `adm/Analysis.py` — split into a namespace facade

Priority: P3

Files touched: `python/pdstools/adm/Analysis.py`,
`python/pdstools/adm/_analysis/` (new package),
`python/tests/adm/test_Analysis.py`,
`python/pdstools/reports/HealthCheckAgent.qmd` (uses `_gini`).

## Problem

`Analysis` (introduced in PR #623, ~1052 lines) groups eight
independent `_check_*` private methods on a single class:

| Method | LOC | Domain |
|---|---|---|
| `_check_channels` | ~100 | channel × direction health |
| `_check_configurations` | ~50 | per-configuration usage |
| `_check_model_maturity` | ~170 | maturity buckets, performance bands |
| `_check_model_performance` | ~30 | overall weighted-avg AUC |
| `_check_taxonomy` | ~100 | actions/treatments/issues/channels counts |
| `_check_response_distribution` | ~100 | Gini of responses/positives |
| `_check_trends` | ~100 | per-snapshot perf trend |
| `_check_predictors` | ~130 | predictor counts, types, performance |
| `_check_predictions` | ~90 | prediction object cross-checks |

The strict AGENTS.md trigger ("namespace facade for >20 public methods")
is **not** met — `Analysis` exposes one public method (`findings()`).
The methods are private orchestration helpers. So this is "would-be-nice"
rather than "must-do".

## Why it was deferred (not done in PR #623's review pass)

Refactor surface area:

- 9 `_check_*` methods × ~100 LOC each → 4 new sub-modules
  (e.g. `channels.py`, `maturity.py`, `taxonomy.py`, `predictors.py`).
- Every sub-module needs its own `Schema`-style `__init__(datamart)` and
  back-reference plumbing.
- 73 tests in `test_Analysis.py` patch via dotted paths like
  `pdstools.adm.Analysis.MetricLimits.evaluate_metric_rag`,
  `pdstools.adm.Analysis.report_utils.n_unique_values`,
  `pdstools.adm.Analysis.cdh_utils.weighted_average_polars`. All those
  patch targets move when the methods relocate, so each test class needs
  its mock paths rewritten.
- `_gini` is imported by `HealthCheckAgent.qmd` from
  `pdstools.adm.Analysis` — needs a re-export shim or a Quarto edit.
- `Finding` dataclass would presumably move to a shared
  `_findings.py` module.

Estimated 3–5 hours of mechanical refactor + test rewiring; the gain is
"dotted name discoverability" (`dm.analysis.maturity._check_*`) which
is currently invisible because the methods are private. No public API
benefit until `findings()` itself is split, which is a separate design
question.

## Proposed approach (when picked up)

Phase 1 — establish the package without behaviour change:
1. Create `python/pdstools/adm/_analysis/` with one module per domain.
2. Move `Finding` and `_gini` into `_analysis/_findings.py`.
3. Each sub-module exposes a class (`ChannelChecks`, `MaturityChecks`,
   …) with `__init__(datamart)` and one `run(...)` method.
4. `Analysis` becomes a 50-line facade that instantiates the sub-checks
   and orchestrates `findings()`.
5. Re-export `Finding`, `_gini` from `pdstools.adm.Analysis` so the
   QMD import keeps working.

Phase 2 — update tests to import from the new locations and patch
narrower mock paths.

## Cross-refs

- AGENTS.md "Namespace facade for large analyzer classes"
- AGENTS.md "Earn your abstractions" (counter-argument: this *might*
  be over-engineering for 9 private helpers)

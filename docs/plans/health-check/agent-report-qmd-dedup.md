# Dedup `HealthCheck.qmd` and `HealthCheckAgent.qmd` via Quarto includes

Priority: P3

Files touched: `python/pdstools/reports/HealthCheck.qmd` (1975 LOC),
`python/pdstools/reports/HealthCheckAgent.qmd` (1734 LOC),
new `python/pdstools/reports/_health_check_*.qmd` partials.

## Problem

`HealthCheckAgent.qmd` was added in PR #623 as a sibling of
`HealthCheck.qmd`, producing a GFM Markdown variant of the same report
for LLM/agent consumption. The two files share a substantial amount of
analysis code — same imports, same datamart load, same per-section
Polars aggregations — but differ in:

- YAML header (`format: html` vs `format: gfm`, theming, CSS)
- Output style (`great_tables.GT` HTML vs plain Markdown tables)
- Plot rendering (Plotly figures vs `print(...)` summaries)
- Several sections in the agent variant deliberately omit visual-only
  content (lift charts, etc.) as part of "reduce overlap with maturity
  report" work in commit `d68adaa5`.

A first-pass head-by-head diff (lines 1–200) shows ~70% of the early
imports/data-loading block could be a shared partial.

## Why it was deferred (not done in PR #623's review pass)

Quarto regressions are easy to ship and hard to test:
- The healthcheck integration tests are Quarto-gated and `@pytest.mark.slow`
  — CI without Quarto installed runs them as `skipped`.
- A subtle params-passing or include-path bug only surfaces when a
  developer actually renders the report locally, which often isn't
  caught until release.
- Quarto `{{< include _x.qmd >}}` directives have rules about chunk
  options (`#| output: false` etc.) and parameter scoping that differ
  from "concatenate the files" intuition.

The PR #623 author already did one pass to "reduce overlap with
maturity report" (commit `d68adaa5`); a second pass to dedup against
`HealthCheck.qmd` is a separate, focused PR.

## Proposed approach (when picked up)

1. Identify the largest shared blocks: imports, datamart loading, the
   per-check section computations (per-channel summary, predictor
   overview, model maturity buckets).
2. Extract each into a `_health_check_<section>.qmd` partial that
   computes data into well-named variables but does **not** render.
3. Each top-level QMD does its own format-specific rendering of those
   variables (`GT(...).as_raw_html()` vs `print(df.to_pandas().to_markdown())`).
4. Run both reports end-to-end against `cdh_sample` and visually diff
   the output before/after.

Risk mitigation: do this as 4–6 small commits (one section at a time),
each with a manual render check.

## Cross-refs

- PR #623 commit `d68adaa5` "reduce overlap with maturity report" —
  prior dedup pass against a different report
- `reports/HealthCheck.qmd`, `reports/HealthCheckAgent.qmd`

# Implementation Plan: Exclusion Rate Distribution (Decision Analyzer)

**Branch**: `feat/da-exclusion-rate` | **Date**: 2026-08-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-optionality-exclusion-rate/spec.md`

## Summary

Add an **exclusion rate** distribution to the Decision Analyzer that mirrors the
existing optionality distribution. The exclusion rate is the per-interaction
fraction of actions lost between a selectable baseline ("from") stage (default
Engagement Policies) and a measurement ("to") stage that follows the existing
optionality stage selector (default Arbitration). Computation reuses the same
per-interaction, per-stage remaining counts that power optionality
(`aggregate_remaining_per_stage`), is exposed as a new `Aggregates` method and a
new `Plot` method with `return_df`, and is surfaced as a new section on the
Optionality Analysis page (Page 7). The section is gated to v2
(decision_analyzer) extracts.

## Technical Context

**Language/Version**: Python >=3.10,<3.15

**Primary Dependencies**: polars (data), plotly (plot), streamlit (app page).
Reuses existing `pdstools.decision_analyzer` machinery; no new third-party deps.

**Storage**: N/A (in-memory LazyFrame over the pre-aggregated Decision Analyzer
views).

**Testing**: pytest — exact-value aggregate tests on the minimal dataset,
plot `return_df`/structure tests, and a Streamlit AppTest state-transition test
for the baseline-stage selector.

**Target Platform**: Local + CI (existing pdstools test matrix); Streamlit app.

**Project Type**: Python library with a Streamlit presentation layer.

**Performance Goals**: Reuse the per-stage remaining counts rather than
recomputing from raw rows; the focused minimal-fixture calculation should stay
within the same order of magnitude as optionality.

**Constraints**: Zero-functionality presentation layer — all computation in the
library, reproducible in a notebook. Follow DecisionAnalyzer namespace/plot
conventions (`return_df`, plots build-and-return figures, no `show`).

**Scale/Scope**: One new aggregate method, one new plot method (+ wiring), one
new app section, and tests. No schema or public-constructor changes.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` is an uninstantiated template (no enforceable
project principles declared), so there are no blocking constitution gates. The
design is nonetheless held to the repo's `AGENTS.md` conventions:

- Computation in the library; the app page only composes widgets and calls
  library methods (zero-functionality presentation layer). ✅
- Plot method returns the figure and accepts `return_df` (no `show`). ✅
- Reuse existing abstractions (`aggregate_remaining_per_stage`,
  `get_optionality_data`) instead of adding a parallel hierarchy. ✅
- Polars expression-based transforms, LazyFrame until a boundary. ✅
- Exact-value tests on the minimal dataset. ✅

## Project Structure

### Documentation (this feature)

```text
specs/002-optionality-exclusion-rate/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── exclusion-rate-contract.md
└── tasks.md   # created by /speckit.tasks (not this command)
```

### Source Code (repository root)

```text
python/pdstools/decision_analyzer/
├── _aggregates.py                 # + get_exclusion_rate_data(...)
└── plots/
    ├── _optionality.py            # + exclusion_rate_distribution(...)
    └── __init__.py                # bind exclusion_rate_distribution onto Plot

python/pdstools/app/decision_analyzer/pages/
└── 7_Optionality_Analysis.py      # + "Exclusion Rate" section (v2 only)

python/tests/decision_analyzer/
├── test_DecisionAnalyzer.py       # exact-value aggregate tests (minimal ds)
└── test_da_plots.py               # plot return_df / structure tests

python/tests/streamlit_apps/decision_analyzer/
└── test_da_pages.py               # page render assertion update
    (+ a focused state-transition AppTest for the baseline selector)
```

**Structure Decision**: Extend the existing `Aggregates` and `Plot` namespaces
and the existing Page 7. No new modules or top-level classes; the metric is a
sibling of optionality and lives beside it.

## Phase 0: Research Plan

1. Confirm the per-interaction, per-stage remaining-count semantics of
   `aggregate_remaining_per_stage` and that optionality's `nOffers` is the exact
   quantity needed for both baseline and measurement counts.
2. Confirm stage ordering / validation via `AvailableNBADStages` and
   `_stage_index`, and the correct default baseline ("Engagement Policies") with
   graceful fallback.
3. Decide the distribution representation (continuous exclusion rate → banded
   histogram) and how `return_df` should shape the data.
4. Confirm v1-vs-v2 gating parity with the optionality funnel and the Page 7
   selector wiring (shared measurement stage).

## Phase 1: Design Plan

1. Design `Aggregates.get_exclusion_rate_data(from_stage, to_stage, df)`
   returning a per-interaction frame (baseline count, measurement count,
   excluded count, exclusion rate), with validation and zero-baseline handling.
2. Design `Plot.exclusion_rate_distribution(from_stage, to_stage, df,
   return_df)` building a percentage-banded distribution figure styled like
   optionality, returning the per-interaction frame when `return_df=True`.
3. Design the Page 7 section: baseline stage selectbox (default Engagement
   Policies), measurement stage sourced from the optionality selector, v2 gating,
   invalid-range warning, filter propagation.
4. Design the test matrix: exact-value aggregate tests (5→2, 4→0, 3→1 ⇒ 60%,
   100%, 66.7%), zero-baseline exclusion, invalid-range error, plot `return_df`
   exactness, and an AppTest selector state-transition.

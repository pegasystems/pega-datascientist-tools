# Decision Analysis roadmap

Open ideas and follow-ups for the Decision Analysis Tool. Priorities are P1
(high), P2 (medium), and P3 (exploratory). Items requiring investigation are
not confirmed defects.

## Correctness and reliability

- **P1 — Mandatory actions:** Decide whether sensitivity, win/loss, and
  optionality should exclude mandatory actions, show them separately, or offer
  a toggle. Preserve their visibility and consider API compatibility; the
  what-if simulations already respect mandatory ranking. Follow-up to #664.
- **P2 — Corrupt uploads:** Report parser errors without crashing Home or
  discarding the previously loaded data; the corrupt-Parquet AppTest is xfailed.
- **P2 — Win/loss accuracy:** Audit Top-N rank propagation and reconcile counts
  across charts and box plots using the same filtered population.
- **P2 — Offer quality edge cases:** Verify no-action, single-action, and
  low-propensity cases and make empty or degenerate results understandable.

## Features and analysis

- **P2 — Data-driven winning ranks:** Replace the current rank-5 ceiling with
  bounds derived from the data, including pre-aggregation and shared controls.
- **P2 — Context filters:** Add Issue and cascading Group filters across pages.
- **P2 — Overview:** Consider a Value Finder-style win/loss proportion view at
  arbitration.
- **P2 — Win/loss deep dives:** Consider separate opponent selectors for wins
  and losses; a general comparison selector already exists.
- **P2 — EE v1 propensities:** Expose model, adjusted, and final propensity
  variants in Arbitration Distribution when available.
- **P2 — Raw interaction history:** Evaluate support and normalization, or
  document the input limitations.
- **P3 — Treatment-level analysis:** Decide how to aggregate and display
  multiple treatments of the same action and which metrics need treatment keys.
- **P3 — Additional views:** Explore winning at a selected rank, overlapping
  funnel filter components, and side-by-side proposition comparisons.
- **P3 — Optionality:** Consider a priority overlay alongside the existing
  propensity line.
- **P3 — A/B results:** Include Impact Analyzer properties when available.

## UI and usability

- **P2 — Large inputs:** Warn about large files and suggest sampling before
  expensive loading; sampling itself already has progress feedback.
- **P2 — Distribution readability:** Add a top-N limiter and sorting for
  crowded Action Distribution charts; check box plots at narrow widths.
- **P2 — Win/loss presentation:** Explore a combined win/loss bar chart and
  use schema-derived property labels where appropriate.
- **P3 — Sample estimates:** Consider showing clearly labeled estimated
  full-population counts rather than only raw sampled counts.

## Architecture and performance

- **P2 — Stages:** Replace remaining hardcoded stage names with data-driven
  stage selection across pages and plots.
- **P2 — Offer quality:** Reuse `aggregate_remaining_per_stage` in
  `get_offer_quality` and keep any remaining analysis logic in the library.
- **P2 — Win/loss internals:** Review overlapping distribution helpers and
  whether cached or pre-aggregated data can replace repeated sample work
  without changing comparison semantics.
- **P3 — Scale:** Benchmark streaming pre-aggregation for large inputs and
  evaluate ingestion of externally pre-aggregated Action Analysis data.
- **P3 — Sampling:** Explore stratified sampling for small channel/direction
  populations.

For implemented cohort handoff guidance, see the
[Decision Analyzer architecture](decision-analyzer-architecture.md#aggregate-summaries-and-row-cohorts).

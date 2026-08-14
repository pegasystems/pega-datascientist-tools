# Phase 0 Research: Exclusion Rate Distribution

## R1 — Per-interaction, per-stage counts (reuse optionality)

**Decision**: Reuse `Aggregates.aggregate_remaining_per_stage` (the same
workhorse behind `get_optionality_data`) to obtain, per interaction, the number
of actions remaining at each stage (`nOffers`). The exclusion rate for an
interaction between a baseline stage `S_from` and a measurement stage `S_to` is:

```
exclusion_rate = (nOffers[S_from] − nOffers[S_to]) / nOffers[S_from]
```

**Rationale**: `aggregate_remaining_per_stage` already computes, for every stage
`S`, the count of actions present at `S` or any later stage — exactly the
"remaining at stage" semantics optionality uses. Building the exclusion rate on
top of the same per-interaction counts guarantees the two metrics are
numerically consistent ("calculated in tandem", per the user) and adds no new
raw-data scan.

**Alternatives considered**:
- Recompute from `remaining_at_stage` row frames and join per interaction —
  correct but duplicates the aggregation optionality already performs and risks
  drift. Rejected.
- Derive from the funnel (`available`/`passing`/`filtered`) — those are
  action-occurrence totals per stage, not per-interaction counts, so they can't
  express a per-interaction distribution. Rejected.

## R2 — Stage selection, defaults, and validation

**Decision**: Accept `from_stage` and `to_stage` as stage names validated
against `AvailableNBADStages` via the existing `_stage_index`. Require
`index(from_stage) <= index(to_stage)`; otherwise raise `ValueError`. Default
baseline in the app is the **Engagement Policies** stage group, with the
existing `stage_selectbox` fallback to the first available stage when absent.

**Rationale**: "Made it past hard eligibility" = after the Eligibility rule in
Engagement Policies, so Engagement Policies is the natural default baseline. The
measurement stage default (Arbitration) is inherited from the optionality
selector, keeping a single control for both metrics (user requirement).
`_stage_index` already raises a clear "Unknown stage …" error, matching the
other stage APIs.

**Alternatives considered**:
- Default baseline "Available Actions": valid but counts the whole catalog,
  including actions removed by hard eligibility, which is *not* what "past hard
  eligibility" means. Offered as a selectable option, not the default.
- Silently clamping an inverted range: hides user error; rejected in favor of a
  warning in the app and a `ValueError` in the library.

## R3 — Distribution representation and `return_df`

**Decision**: The exclusion rate is continuous on [0, 1]. The library method
returns a **per-interaction** frame; the plot buckets it into 20 fixed
5-percentage-point bands and shows the **share of interactions** per band
(percentage y-axis), with a green-to-red bar scale from low to high exclusion.
When available, an optionality-style propensity line is placed on a secondary
axis but hidden by default.
`return_df=True` returns the per-interaction frame (the data that drives the
chart), per the repo's plot conventions.

**Rationale**: Returning per-interaction values keeps the metric exact and
scriptable (each interaction's rate is verifiable), and lets the plot own the
purely-visual binning. This matches the AGENTS.md rule that `return_df` returns
"the underlying (Lazy)Frame that drives the chart".

**Alternatives considered**:
- Returning pre-bucketed band counts from the aggregate: couples the library to
  a presentation choice (bin edges) and makes exact per-interaction assertions
  impossible. Rejected — binning stays in the plot layer.
- Box/violin instead of histogram: less directly comparable to the optionality
  bars the user referenced ("in a way like we do for optionality today").
  Histogram chosen; a box view can be a later enhancement.

## R4 — v1/v2 gating and Page 7 wiring

**Decision**: Render the Exclusion Rate section only for
`extract_type == "decision_analyzer"` (v2). The measurement stage is read from
the existing optionality stage selector session key; the baseline stage gets its
own selectbox. Data is the page's already-filtered frame
(`decision_data.filtered(collect_page_filters())`).

**Rationale**: v1 (explainability_extract) only has synthetic Arbitration/Output
stages, so a pre-arbitration baseline is undefined — the same reason the
optionality funnel is v2-only. Sharing the optionality selector for the "to"
stage matches the user's instruction ("the current optionality distribution plot
has a selector, that should be the same for the 'to'"). Reusing the page filter
keeps the section consistent with the rest of Page 7.

**Alternatives considered**:
- Supporting v1 with from=Arbitration,to=Output: technically possible but not
  the user's intent (past-eligibility→arbitration) and of little analytic value.
  Rejected.
- A second independent "to" selector: contradicts the shared-selector
  requirement and adds redundant UI. Rejected.

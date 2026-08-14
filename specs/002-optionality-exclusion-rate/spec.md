# Feature Specification: Exclusion Rate Distribution (Decision Analyzer)

**Feature Branch**: `feat/da-exclusion-rate`

**Created**: 2026-08-14

**Status**: Draft

**Input**: User description: "Report optionality in the app as an exclusion rate. Exclusion rate is the percentage of offers that made it past hard eligibility criteria but drop off before eligibility [arbitration]. It is very much related to optionality and could perhaps be calculated in tandem. We'll need an additional definition of the stage 'from where' to count but the 'arbitration' stage is I think the same. I'd like to report this exclusion rate as a distribution plot, in a way like we do for optionality today."

## Context

The Decision Analyzer "Optionality Analysis" page (Page 7) shows how many
actions remain per customer at a chosen stage (default **Arbitration**). It is
essentially a *survival* view: how much choice is left.

The **exclusion rate** is the complementary *attrition* view: of the actions a
customer had at an earlier baseline stage, what fraction were **excluded**
before reaching the measurement stage. Where optionality answers "how much
choice remains?", exclusion rate answers "how much choice was lost along the
way?". The two are computed from the same per-interaction, per-stage remaining
counts and are therefore calculated in tandem.

**Hard eligibility** in NBAD is the Eligibility rule inside the *Engagement
Policies* stage group. "Made it past hard eligibility" therefore means the
default baseline ("from") stage is the output of Engagement Policies, while the
measurement ("to") stage matches the optionality selector (default
Arbitration).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See how many actions are excluded before arbitration (Priority: P1)

As a decision-strategy analyst, I want to see the distribution of the
per-customer exclusion rate between a baseline stage and the arbitration stage,
so I can quantify how aggressively engagement policies and downstream filters
prune the action set before prioritization.

**Why this priority**: This is the core value of the feature — a single,
readable distribution that reveals whether customers are being over-filtered
(little left to arbitrate) versus lightly filtered. It reuses the optionality
data already computed, so it is a high-value, low-cost addition.

**Independent Test**: Load a v2 (decision_analyzer) extract, open Page 7, and
observe a new "Exclusion Rate" distribution showing the percentage of
interactions in each exclusion-rate band between the default baseline
(Engagement Policies) and the optionality stage (Arbitration).

**Acceptance Scenarios**:

1. **Given** a v2 extract with actions filtered between Engagement Policies and
   Arbitration, **When** the analyst views the Exclusion Rate section, **Then**
   a distribution plot shows the share of interactions per exclusion-rate band,
   with 0% meaning no actions were lost and 100% meaning all were lost.
2. **Given** an interaction that had 5 actions at the baseline stage and 2 at
   the measurement stage, **When** its exclusion rate is computed, **Then** it
   equals 60%.

---

### User Story 2 - Override the baseline and measurement stages (Priority: P2)

As an analyst, I want to change the "from" (baseline) stage and have the "to"
(measurement) stage follow the existing optionality stage selector, so I can
inspect exclusion between any two points of the pipeline (e.g. Available
Actions → Arbitration, or Engagement Policies → Output).

**Why this priority**: Analysts investigate different pipeline segments. A fixed
pair would only answer one question; a selector makes the feature reusable
without cluttering the page.

**Independent Test**: On Page 7, change the baseline stage selector and confirm
the exclusion-rate distribution updates; change the optionality stage selector
and confirm the exclusion-rate measurement point follows it.

**Acceptance Scenarios**:

1. **Given** the baseline selector is set to "Available Actions" and the
   optionality/measurement stage is "Arbitration", **When** the analyst views
   the plot, **Then** exclusion is computed from Available Actions to
   Arbitration.
2. **Given** the analyst selects a baseline that is *after* the measurement
   stage, **When** the section renders, **Then** a clear warning is shown
   instead of a misleading plot.

---

### User Story 3 - Read exclusion rate alongside optionality (Priority: P3)

As an analyst, I want the exclusion-rate distribution presented next to the
optionality distribution on the same page, so I can reason about survival and
attrition together without switching context.

**Why this priority**: Co-location aids interpretation but is not required for
the metric to deliver value; it is a presentation refinement.

**Independent Test**: Confirm the Exclusion Rate section renders on Page 7
below the existing Optionality content and respects the same global and
contextual (channel) filters.

**Acceptance Scenarios**:

1. **Given** a channel filter is applied on Page 7, **When** the analyst views
   the Exclusion Rate section, **Then** the distribution reflects only the
   filtered interactions.

---

### Edge Cases

- **Interaction with zero actions at the baseline stage**: no exclusion rate can
  be defined (division by zero). Such interactions are excluded from the
  distribution.
- **Interaction whose actions all survive**: exclusion rate is 0%.
- **Interaction whose actions are all removed**: exclusion rate is 100%.
- **Baseline stage == measurement stage**: exclusion rate is 0% for every
  interaction (nothing can be lost across a zero-length segment).
- **Baseline stage later than measurement stage**: invalid; the system reports a
  clear error/warning rather than a negative or misleading rate.
- **v1 (explainability_extract) data**: only synthetic Arbitration/Output stages
  exist, so a pre-arbitration baseline is not available. The exclusion-rate
  section is not shown for v1, mirroring the existing optionality-funnel gating.
- **Empty filtered dataset** (e.g. a channel with no data): the section shows
  the same "no data" handling as the rest of the page.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST compute, per interaction, an exclusion rate equal
  to `(actions_at_baseline − actions_at_measurement) / actions_at_baseline`,
  where `actions_at_stage` is the number of that interaction's actions remaining
  at (or after) the given stage — the same per-interaction, per-stage counts
  that drive optionality.
- **FR-002**: The system MUST allow the baseline ("from") stage to be selected,
  defaulting to the Engagement Policies stage group ("made it past hard
  eligibility"), falling back to the first available stage when that default is
  absent from the data.
- **FR-003**: The measurement ("to") stage MUST follow the existing optionality
  stage selector on Page 7 (default Arbitration), so a single control governs
  both the optionality and exclusion-rate measurement points.
- **FR-004**: The system MUST present the exclusion rate as a distribution plot
  over interactions (share of interactions per exclusion-rate band), styled
  consistently with the existing optionality distribution.
- **FR-005**: Interactions with zero actions at the baseline stage MUST be
  excluded from the exclusion-rate distribution.
- **FR-006**: When the selected baseline stage is later in the pipeline than the
  measurement stage, the system MUST surface a clear warning and MUST NOT
  render a misleading distribution.
- **FR-007**: The exclusion-rate computation MUST respect the page's global and
  contextual (channel) filters, consistent with the optionality section.
- **FR-008**: The exclusion-rate section MUST only render for v2
  (decision_analyzer) extracts, where real pre-arbitration pipeline stages
  exist; it MUST NOT appear for v1 (explainability_extract) extracts.
- **FR-009**: The library method producing the plot MUST accept a
  `return_df` option that returns the underlying per-interaction frame instead
  of the figure, so the metric is scriptable and testable outside the app.
- **FR-010**: Unknown or invalid stage names passed to the computation MUST
  raise a clear, actionable error (consistent with existing stage APIs).

### Key Entities *(include if feature involves data)*

- **Per-interaction exclusion record**: one row per interaction that has at least
  one action at the baseline stage. Attributes: interaction identifier, action
  count at baseline stage, action count at measurement stage, number of actions
  excluded, exclusion rate (0–1). Derived from the per-interaction, per-stage
  remaining counts shared with optionality.
- **Exclusion-rate distribution**: aggregation of exclusion records into 20
  5-percentage-point bands from `0-5%` through `95-100%`, with the share of
  interactions in each band. Bars use a green-to-red scale from low to high
  exclusion. When usable propensity data exists, an optionality-style line is
  available on a secondary axis and hidden by default.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a known dataset, an analyst can read the percentage of
  interactions whose action set was reduced by more than a given threshold
  (e.g. ">50% excluded") directly from the distribution.
- **SC-002**: The per-interaction exclusion rate is exactly reproducible outside
  the app (via the library method's `return_df`) and matches hand-computed
  values on a minimal dataset (e.g. 60%, 100%, 66.7% for baseline→measurement
  counts of 5→2, 4→0, 3→1).
- **SC-003**: Changing the baseline or measurement stage updates the
  distribution to reflect the newly selected pipeline segment.
- **SC-004**: The exclusion-rate calculation reuses the existing per-stage
  aggregation path rather than performing a separate raw-data scan; on the
  minimal fixture, its focused calculation completes within the same order of
  magnitude as the existing optionality calculation.

## Assumptions

- The exclusion rate is a per-interaction attrition measure summarized as a
  distribution across interactions (mirroring how optionality is presented),
  not a single global percentage.
- "Actions remaining at a stage" uses the same semantics as optionality: an
  action counts at a stage if it is present at that stage or any later stage.
- The measurement stage is shared with (driven by) the existing optionality
  stage selector; only the baseline stage introduces a new control.
- The feature targets v2 (decision_analyzer) extracts; v1 lacks the
  pre-arbitration stages needed for a meaningful baseline.
- Presentation lives on the existing Optionality Analysis page (Page 7); all
  computation lives in the library so it is reproducible in a notebook.

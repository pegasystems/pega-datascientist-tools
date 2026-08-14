# Phase 1 Data Model: Exclusion Rate Distribution

All frames are polars; the aggregate returns a `LazyFrame` (collected only at
the app/plot boundary).

## Entity: Per-interaction exclusion record

Produced by `Aggregates.get_exclusion_rate_data(from_stage, to_stage, df)`. One
row per interaction that has **at least one action at the baseline stage**.

| Column           | Type    | Description                                                             |
| ---------------- | ------- | ----------------------------------------------------------------------- |
| `Interaction ID` | Utf8    | Interaction identifier.                                                 |
| `Actions From`   | UInt/Int| Actions remaining at the baseline (`from`) stage (`nOffers[S_from]`).    |
| `Actions To`     | UInt/Int| Actions remaining at the measurement (`to`) stage (`nOffers[S_to]`).     |
| `Excluded`       | Int     | `Actions From − Actions To` (≥ 0 given a valid, ordered stage pair).     |
| `Exclusion Rate` | Float64 | `Excluded / Actions From`, in `[0, 1]`.                                  |

**Invariants**:
- `Actions From ≥ 1` for every row (zero-baseline interactions are dropped).
- `0 ≤ Actions To ≤ Actions From` for a valid ordered stage pair.
- `0.0 ≤ Exclusion Rate ≤ 1.0`.
- `from_stage == to_stage` ⇒ `Excluded == 0` and `Exclusion Rate == 0.0`.

**Validation & errors**:
- `from_stage` / `to_stage` not in `AvailableNBADStages` ⇒ `ValueError`
  (via `_stage_index`).
- `index(from_stage) > index(to_stage)` ⇒ `ValueError` (inverted range).

## Entity: Exclusion-rate distribution (plot-side)

Derived inside `Plot.exclusion_rate_distribution` by binning `Exclusion Rate`
into 40 fixed 2.5-percentage-point bands and normalizing to the share of
interactions. Bars are shaded green at low exclusion and red at high exclusion.

| Field            | Type    | Description                                            |
| ---------------- | ------- | ------------------------------------------------------ |
| band             | category| One of 40 bands from `0-2.5%` through `97.5-100%`.     |
| pct_interactions | Float64 | Share of interactions falling in the band (sums ~100%).|

This entity is internal to the plot; `return_df=True` returns the
**per-interaction record** above (not the banded frame), so callers can bin or
summarize however they like.

## Relationship to optionality

`get_optionality_data` and `get_exclusion_rate_data` are siblings computed from
the same `aggregate_remaining_per_stage` per-interaction, per-stage counts:

- Optionality reads `nOffers` at a single stage → distribution of *remaining*
  actions.
- Exclusion rate reads `nOffers` at two stages → distribution of the *fraction
  lost* between them.

## Worked example (minimal dataset, from = Eligibility, to = Output)

Per-interaction remaining counts in `data/da/sample_eev2_minimal.csv`:

| Interaction | Actions From (Eligibility) | Actions To (Output) | Excluded | Exclusion Rate |
| ----------- | -------------------------- | ------------------- | -------- | -------------- |
| INT-001     | 5                          | 2                   | 3        | 0.60           |
| INT-002     | 4                          | 0                   | 4        | 1.00           |
| INT-003     | 3                          | 1                   | 2        | 0.6667         |

These exact values anchor the aggregate unit tests.

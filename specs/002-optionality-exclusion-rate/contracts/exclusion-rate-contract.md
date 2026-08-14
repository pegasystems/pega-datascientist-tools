# Contract: Exclusion Rate library API

Public library surface added by this feature. The Streamlit page consumes only
these methods (zero-functionality presentation layer).

## `Aggregates.get_exclusion_rate_data`

```python
def get_exclusion_rate_data(
    self,
    from_stage: str,
    to_stage: str = "Arbitration",
    df: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    ...
```

**Parameters**
- `from_stage`: baseline stage name (must be in `AvailableNBADStages`).
- `to_stage`: measurement stage name (default `"Arbitration"`); must be in
  `AvailableNBADStages` and at or after `from_stage` in pipeline order.
- `df`: input frame; defaults to `self.da.sample`.

**Returns**: `LazyFrame` with columns `Interaction ID`, `Actions From`,
`Actions To`, `Excluded`, `Exclusion Rate` — one row per interaction with
`Actions From ≥ 1` (see data-model.md).

**Raises**: `ValueError` for unknown stages or an inverted stage range
(`from_stage` later than `to_stage`).

**Guarantees**
- Exclusion rate is exact and in `[0, 1]`.
- Interactions with zero actions at `from_stage` are omitted.
- Respects any filters already applied to `df`.

## `Plot.exclusion_rate_distribution`

```python
def exclusion_rate_distribution(
    self,
    from_stage: str,
    to_stage: str = "Arbitration",
    df: pl.LazyFrame | None = None,
    return_df: bool = False,
):
    ...
```

**Behavior**
- Builds a distribution figure with 40 2.5-percentage-point bands on the x-axis,
  from `0-2.5%` through `97.5-100%`, and share of interactions on the y-axis.
  Bars use a green-to-red color scale from low to high exclusion and follow the
  Pega template used by `propensity_vs_optionality`.
- `return_df=True` returns the per-interaction `LazyFrame` from
  `get_exclusion_rate_data` (the data that drives the chart) instead of the
  figure.
- Never calls `fig.show()`; the caller decides how to display (AGENTS.md).

**Raises**: propagates `ValueError` from `get_exclusion_rate_data`.

## App contract (Page 7 — Optionality Analysis)

- A new "Exclusion Rate" section renders **only** for
  `extract_type == "decision_analyzer"` (v2).
- Baseline stage: a `stage_selectbox` defaulting to `"Engagement Policies"`
  (falls back to the first available stage when absent), stored under a distinct
  session key (e.g. `exclusion_from_stage`).
- Measurement stage: read from the existing optionality stage selector session
  key (`optionality_stage`, default `"Arbitration"`).
- If the selected baseline is later than the measurement stage, show a
  `st.warning` and skip the plot.
- Uses the page's filtered frame (`decision_data.filtered(collect_page_filters())`),
  matching the optionality section.
- Renders via `st.plotly_chart(...)` with no deprecated width args.

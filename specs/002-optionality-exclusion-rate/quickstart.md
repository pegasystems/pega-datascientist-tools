# Quickstart: Exclusion Rate Distribution

Validation scenarios proving the feature works end-to-end. See
[contracts/exclusion-rate-contract.md](./contracts/exclusion-rate-contract.md)
for the API surface and [data-model.md](./data-model.md) for the data shapes.

## Prerequisites

```bash
uv sync --extra tests
```

## Scenario 1 — Exact per-interaction exclusion (library, minimal dataset)

Reproduces the worked example (baseline = Eligibility, measurement = Output).

```python
import polars as pl
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer

da = DecisionAnalyzer(pl.scan_csv("data/da/sample_eev2_minimal.csv"), sample_size=5000)

df = (
    da.aggregates.get_exclusion_rate_data(from_stage="Eligibility", to_stage="Output")
    .collect()
    .sort("Interaction ID")
)
print(df.select("Interaction ID", "Actions From", "Actions To", "Excluded", "Exclusion Rate"))
```

**Expected**: INT-001 → 5/2/3/0.60, INT-002 → 4/0/4/1.00, INT-003 → 3/1/2/0.6667.
INT-002 is the only 100%-excluded interaction; no interaction is dropped for a
zero baseline (all have ≥1 action at Eligibility).

## Scenario 2 — Invalid stage range raises

```python
da.aggregates.get_exclusion_rate_data(from_stage="Output", to_stage="Eligibility")
# → ValueError (from_stage is later than to_stage)
```

## Scenario 3 — Plot returns a figure and `return_df` returns the frame

```python
fig = da.plot.exclusion_rate_distribution(from_stage="Eligibility", to_stage="Output")
assert fig.data  # a Plotly figure with at least one trace
assert len(fig.data[0].x) == 20  # 5% bands from 0-5% through 95-100%
assert fig.data[0].marker.color[0] != fig.data[0].marker.color[-1]

frame = da.plot.exclusion_rate_distribution(
    from_stage="Eligibility", to_stage="Output", return_df=True
).collect()
assert set(frame.columns) >= {"Interaction ID", "Exclusion Rate"}
```

## Scenario 4 — App section (Streamlit Page 7)

```bash
uv run pdstools --help    # confirm the decision-analyzer app entry point
```

Launch the Decision Analysis app, load a **v2** extract, open **Optionality
Analysis**, and confirm:

- An "Exclusion Rate" section appears below the optionality content.
- A baseline ("from") stage selector defaults to Engagement Policies.
- The measurement point follows the existing optionality stage selector.
- Selecting a baseline later than the measurement stage shows a warning instead
  of a plot.
- For a **v1** extract, the section does not appear.

## Test commands

```bash
# Exact-value aggregate tests
uv run pytest python/tests/decision_analyzer/test_DecisionAnalyzer.py -k "exclusion" -q

# Plot tests
uv run pytest python/tests/decision_analyzer/test_da_plots.py -k "exclusion" -q

# App page tests
uv run pytest python/tests/streamlit_apps/decision_analyzer/test_da_pages.py -q
```

# Decision Analyzer

Decision Analyzer in `pdstools` helps you analyze Next-Best-Action decisioning data from Explainability Extract exports.
It is built to explore the decision funnel end-to-end, understand arbitration behavior, and generate insights from real-time decision events.

The Decision Analyzer example notebook (`examples/decision_analyzer/decision_analyzer.ipynb`) and Explainability Extract notebook (`examples/explainability_extract/explainability_extract.ipynb`) use this module directly.
In practice, those notebooks are guided walkthroughs of the same `DecisionAnalyzer` class and utilities defined here.

## Core engine (`decision_data.py`)

The central implementation lives in `decision_data.py`.
`DecisionAnalyzer` handles the end-to-end analysis flow:

- Detects source type (`eev1`/`eev2`) and selects the matching table definition
- Validates and standardizes columns and data types
- Cleans and enriches raw records (for example `pxRank`, stage fields, and day-level fields)
- Applies global filters and invalidates/rebuilds cached views safely
- Builds sampled and pre-aggregated datasets used by notebook analyses and app pages

## Schema mapping (`table_definition.py`)

`table_definition.py` defines the expected schema for both extract types.
It is used for mapping the column names in the raw data into understandable column names used in the code.

Example mappings (raw source -> internal name):

- `Primary_ContainerPayload_Channel` -> `pyChannel`
- `Stage_pyStageGroup` -> `StageGroup`

At runtime, `DecisionAnalyzer` uses this mapping to:

- Select the correct table definition for the detected extract type
- Rename source columns to the internal canonical names used by analyses
- Cast columns to expected Polars types before cleanup and aggregation
- Keep only relevant columns for the downstream analysis flow

## What it supports

- Aggregation and analysis across stages in the decision funnel
- Arbitration-focused diagnostics and visual exploration
- Flexible filtering and slicing by issue, group, action, channel, and stage

## Supported data sources (`eev1` and `eev2`)

The `DecisionAnalyzer` class accepts Explainability Extract data from:

- `eev1`: Arbitration-stage-focused extract
- `eev2`: Full decision-funnel extract (all stages)

Source type is determined automatically by `determine_extract_type` in `utils.py`:

- If column `pxStrategyName` is present, data is treated as `decision_analyzer` (full funnel / `eev2` style)
- Otherwise, data is treated as `explainability_extract` (`eev1` style)

This automatic detection selects the appropriate table definition and processing path internally.

## Decision Analyzer app

This project also includes a Streamlit app in `python/pdstools/app/decision_analyzer`.
The app uses the same `DecisionAnalyzer` class from this module as its analysis engine and provides a UI for loading data, applying filters, and exploring funnel and arbitration views.

## Quick start

```python
import polars as pl
from pdstools.decision_analyzer import DecisionAnalyzer

raw_data = pl.scan_parquet("path/to/explainability_extract.parquet")
analyzer = DecisionAnalyzer(raw_data)
```

From there, call `DecisionAnalyzer` methods such as `getFunnelData()` and `getDistributionData()` to access aggregated data views for analysis.
For visualizations, use `analyzer.plot` methods such as `decision_funnel()` and `distribution_as_treemap()`.

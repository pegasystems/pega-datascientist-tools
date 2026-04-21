# Move report-local plot helpers into `adm/Plots/ (package)`

**Priority:** P3
**Touches:** `reports/ModelReport.qmd`, `python/pdstools/adm/Plots/ (package)`, `python/pdstools/adm/BinAggregator.py`

`ModelReport.qmd` defines several Plotly figures inline:
- Cumulative gains/lift area charts
- Base-propensity hline overlay on predictor binning
- The "Philip Mann" lift bar plot (also duplicated in `BinAggregator.py`)

Inline definitions are untestable and not reusable from other reports.

## Approach

Lift these into reusable functions on `ADMDatamart.plot` (or a dedicated `ModelReport` helper class). Add tests. Remove the duplication between `ModelReport.qmd` and `BinAggregator.py`.

# Move inline plots to `plots` module (Page 9 — Thresholding Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py`, `plots.py`

Three Plotly charts are defined inline in the Thresholding page rather than in the shared `plots.py` module.

## Approach

Extract the three inline chart definitions into reusable functions in `plots.py`. Add parameter-level tests.

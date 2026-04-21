# Consistent stage color scheme (Page 7 — Optionality Analysis)

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py`, `plots.py`

Stage colours are chosen independently on each page.

## Approach

Define a single shared stage colour palette (in `da_streamlit_utils.py` or `plots.py`) and use it across all DA pages.

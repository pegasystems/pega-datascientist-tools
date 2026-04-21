# Reduce session-state footprint (Page 8 — Offer Quality Analysis)

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/pages/8_Offer_Quality_Analysis.py`

Page 8 stores more in `st.session_state` than necessary (full data frames, intermediate results).

## Approach

Trim session state to derived values and UI state only. Recompute data from the cached `DecisionAnalyzer` instance as needed.

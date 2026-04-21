# Scale up counts after sampling

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`, Streamlit UI

The sampling infrastructure (sample fraction metadata) is in place but UI still shows raw sampled counts rather than estimated full-volume counts.

## Approach

Use the stored sample fraction to scale raw counts back to estimated full-population values for display. Show a "~" prefix or tooltip to indicate estimates.

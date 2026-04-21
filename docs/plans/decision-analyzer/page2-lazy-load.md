# Speed up first-use page load (Page 2 — Overview)

**Priority:** P1
**Touches:** `python/pdstools/app/decision_analyzer/pages/2_Overview.py`

Page 2 computes all stage-specific metrics upfront before rendering any tab. First load is slow for large datasets.

## Approach

Lazy-load stage-specific metrics per tab — only compute what the currently visible tab needs. Use `st.tabs` lazy evaluation or compute on tab click.

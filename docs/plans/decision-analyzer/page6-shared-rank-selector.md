# Shared rank selector component (Page 6 — Win/Loss Analysis)

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`, pages 5, 6, and others

Rank selection UI is duplicated across Win/Loss, Sensitivity, and other pages.

## Approach

Extract a `rank_selector(key, label, ...)` helper in `da_streamlit_utils.py` and replace inline selectors across pages.

# Add Issue filter to contextual filters section

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`, all DA pages

The contextual filters section currently lacks an Issue multiselect, limiting how users can scope analysis.

## Approach

- Extend `contextual_filters()` with an `issue_selector()` multiselect (empty = no filter).
- Store in `page_issue_filter` / `page_issue_expr` session state keys (same pattern as channel).
- Append the new key to the tuple in `collect_page_filters()` so it flows through `DecisionAnalyzer.filtered`.
- Update empty-data checks on all pages.

**Depends on:** nothing (can be done independently)

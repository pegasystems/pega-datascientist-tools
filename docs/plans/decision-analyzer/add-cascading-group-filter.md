# Add cascading Group filter to contextual filters section

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`, all DA pages
**Depends on:** `add-issue-filter`

After the Issue filter is in place, add a Group filter that dynamically constrains its options based on the current Issue selection (Issue → Group hierarchy).

## Approach

- Add `group_selector()` multiselect that re-queries available groups based on the active Issue selection.
- Same session state pattern (`page_group_filter` / `page_group_expr`).
- Add the key to `collect_page_filters()`; pages already pick up new keys automatically via `da.filtered(collect_page_filters())`.

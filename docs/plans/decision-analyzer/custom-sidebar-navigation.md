# Custom sidebar navigation with disabled pages

**Priority:** P1
**Touches:** `python/pdstools/app/decision_analyzer/`

Streamlit hides unavailable pages entirely. Users have no way to know a page exists until prerequisites are met.

## Approach

- Define a page registry with prerequisites and availability predicates.
- Render all pages in the sidebar; grey out unavailable ones with a reason message (e.g. "Upload data first").
- Add `ensure_page_access()` guard at the top of each page so direct navigation is also blocked gracefully.

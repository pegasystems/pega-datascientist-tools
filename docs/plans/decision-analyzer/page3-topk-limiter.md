# Top-K limiter (Page 3 — Action Distribution)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/3_Action_Distribution.py`

Bar charts with more than ~50 actions become unreadable.

## Approach

Add a "Show top N" control with sorting options (by count, by name, etc.). Default to top 20. Allow users to override.

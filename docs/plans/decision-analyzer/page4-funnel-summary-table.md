# Funnel summary table improvements (Page 4 — Action Funnel)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/4_Action_Funnel.py`

Current table shows averages/percentages only, making it hard to gauge absolute impact.

## Approach

- Add absolute-count display mode toggle.
- Include a synthetic "Available Actions" baseline row.
- Clarify column headers.

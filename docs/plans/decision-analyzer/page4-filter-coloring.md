# Better coloring for filter components (Page 4 — Action Funnel)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/4_Action_Funnel.py`, `plots.py`

Filter components currently use arbitrary colours; related components (eligibility, suitability, engagement) should share a color family.

## Approach

Use a consistent color scheme by filter category. Map eligibility/suitability/engagement to distinct color families using `pdstools/utils/color_mapping.py`.

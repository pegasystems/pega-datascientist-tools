# Dynamic stage UI

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/`, `plots.py`

Hardcoded stage names remain in the Offer Quality page
(`stage="Output"` in `pages/8_Offer_Quality_Analysis.py`) and in
`decision_analyzer/plots.py` (`"Final"` filter, `stage="Final"`
default on `action_variation`). This affects pages 3, 7, 8, 10 and
the shared `plots.py`.

## Approach

Replace hardcoded stage names with data-driven
`DecisionAnalyzer.stages` (or the more specific
`stages_from_arbitration_down` / `stages_with_propensity` accessors)
throughout. Audit all pages and plot functions for hardcoded stage
references.

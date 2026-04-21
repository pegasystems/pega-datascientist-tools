# Support propensity-based categories per stage (Page 8 — Offer Quality Analysis)

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/pages/8_Offer_Quality_Analysis.py`, `plots.py`

Some stages can categorise actions by propensity bucket; this is not currently surfaced.

## Approach

Where propensity-based categorisation is meaningful for a stage, expose it as a coloring/faceting option on the offer quality charts.

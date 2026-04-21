# Move offer quality logic to DecisionAnalyzer (Page 8 — Offer Quality Analysis)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/8_Offer_Quality.py`, `decision_analyzer.py`

Offer quality calculations are currently embedded in the Streamlit page code, making them untestable and non-reusable.

## Approach

Extract into `DecisionAnalyzer` (as a method or property). Add unit tests. Page becomes a thin rendering layer.

**Related:** `refactor-offer-quality.md` (core library cleanup of `get_offer_quality`).

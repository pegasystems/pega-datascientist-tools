# EE v1: show all propensity flavors (Page 10 — Arbitration Component Distribution)

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/pages/10_Arbitration_Distribution.py`

EE v1 data has model propensity, adjusted propensity, and final propensity. The page currently only displays one.

## Approach

Detect EE v1 format and surface all three propensity flavors (e.g. as separate tabs or a selector).

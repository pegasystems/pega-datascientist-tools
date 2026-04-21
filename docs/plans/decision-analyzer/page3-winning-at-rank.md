# "Winning at rank X" view (Page 3 — Action Distribution)

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/pages/3_Action_Distribution.py`, `plots.py`

Currently shows win frequency only at the final stage. Users want to know how often each action wins at a given rank.

## Approach

Add a "Winning at rank X" view: plot how often each action achieves rank ≤ X, with X selectable by the user.

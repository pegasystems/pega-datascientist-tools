# Drill Down — frame cannibalization at action level, not channel

**Priority:** P1
**Touches:** `python/pdstools/app/impact_analyzer/pages/2_Drill_Down.py`,
`python/pdstools/impactanalyzer/ImpactAnalyzer.py`

The "Lift Cannibalization" section currently explains the phenomenon at
channel level (NBA redistributes impressions across channels, so a
channel can show negative lift while overall lift is positive). The
practical concern in NBA tuning is at *action* level: NBA boosting a
high-CTR action can starve another action that the control group would
have served. Channel-level cannibalization is real but secondary.

## Approach

- Confirm whether the current PDC/VBD aggregates carry per-action
  granularity. PDC exports often aggregate to channel/issue/group; VBD
  retains action.
- If action-level data is available in either format, add an
  action-level cannibalization view (per-action lift, with the same
  positive-overall / negative-per-action detection logic).
- Re-word the existing copy to make clear that the channel-level view is
  one slice; action-level is the more common diagnostic.
- See related plan `drill-overview-always-by-channel.md` (the channel
  view stays as the default overview).

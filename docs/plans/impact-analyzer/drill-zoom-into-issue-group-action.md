# Channels — zoom further into Issue / Group / Action

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/pages/2_Channels.py`,
`python/pdstools/impactanalyzer/ImpactAnalyzer.py`

The Channels page currently only zooms in along the **Channel** axis.
The Pega product UI also lets users drill into **Issue**, **Group**, and
**Action**. Adding the same axes here would bring the open-source tool
to parity with product.

## Approach

- Generalise `summarize_experiments(by=...)` callers so the page can pivot
  on any of `Channel`, `Issue`, `Group`, `Name` (Action) — whichever
  dimensions are present in the loaded `ia_data`.
- Add a "Drill axis" selector at the top of the page (default: Channel).
  The rest of the page (overview metrics, per-slice cards, redistribution
  chart, detailed table) re-renders against the chosen axis.
- For Issue / Group / Action, confirm the input source actually carries
  the column. PDC exports often aggregate to channel/issue/group; VBD
  retains action. Skip axes whose column is absent or all-null.

## Action-level cannibalization warnings

When the drill axis is **Action** (and arguably **Group**), the
cannibalization framing that we removed from the channel-level view in
PR #796 *is* appropriate: NBA boosting a high-CTR action can starve
another action that the control group would have served, so per-action
lift can be negative even when overall lift is positive (Simpson's
paradox).

When implementing this feature, bring back the cannibalization warning
banner + per-slice "negative lift while overall is positive" callout +
"Why does action cannibalization happen?" educational expander, but
gate them on `drill_axis in {"Name", "Group"}`. The original copy and
CSS class (`.cannib-warning`) are in PR #796's diff if a starting
point is useful.

Cross-ref: `drill-cannibalization-action-level.md` (the existing P1
plan item that this would resolve in passing).

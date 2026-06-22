# Show absolute counts alongside percentages in hover tooltips

**Priority:** P2
**Touches:**
- `python/pdstools/decision_analyzer/plots/_optionality.py`
- `python/pdstools/decision_analyzer/plots/_winloss.py`
- `python/pdstools/decision_analyzer/plots/_funnel.py` (decisions-without-actions trace)

Several charts show only percentages in their hover tooltips, making it hard to judge
whether a group is large or negligible in absolute terms. This matters especially when
deciding whether a customer sub-group description is worth adding — a group that is
"20% of decisions" is much more actionable when you can immediately see it represents
50 000 vs. 12 interactions.

Gaps identified:

| Plot | Current hover | Missing |
|---|---|---|
| Optionality histogram | `Decisions = X%` | absolute decision count |
| Optionality scatter | `Avg Propensity = X%` | decision count for that optionality bucket |
| Win/Loss pie charts | `X%` | absolute win/loss count |
| Decisions-without-actions bar | `X% of decisions` | absolute interaction count |

(The funnel reach chart and the sensitivity chart already show both absolute counts and
percentages, so they do not need changes.)

## Approach

Use Plotly's `customdata` / `hovertemplate` mechanism (already used elsewhere in the
same files):

1. **Optionality histogram** — add the raw decision count to `customdata` and extend
   `bar_hovertemplate` to `"Optionality = %{x}<br>Decisions = %{y:.1f}% (%{customdata[N]:,})<extra></extra>"`.
2. **Win/Loss pies** — replace `%{percent}` with
   `"%{percent} (%{value:,})<extra></extra>"` so the raw count appears next to the share.
3. **Decisions-without-actions bar** — pass the raw interaction count as `customdata`
   and extend the hover template to `"%{y:.1f}% of decisions (%{customdata:,})<extra></extra>"`.

# Mandatory actions: how should sensitivity / win-loss treat them?

**Priority:** P1 (follow-up to #664, phases 1+2 already shipped)
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`,
`python/pdstools/decision_analyzer/plots.py`, Streamlit pages 5 & 6.

## Background

Actions with priority ≥ `MANDATORY_PRIORITY_THRESHOLD` (5M) bypass normal
arbitration and are always slotted in the top rank. As of #664 phases 1+2
the analyzer auto-detects these rows (`is_mandatory = 1`) and the win-loss
pie marks them with a "★" prefix plus a callout. The underlying analysis
math is unchanged — mandatory rows still flow through sensitivity, win/loss
distribution, and optionality calculations exactly like ordinary rows.

That is silently wrong: any "win rate" for a mandatory action is
tautologically high (it never lost arbitration), and the sensitivity of
the priority factor is dominated by these rows in deployments that use
mandatory actions heavily.

## Open question

How should sensitivity / win-loss / optionality treat mandatory rows?

Three options on the table:

1. **Exclude mandatory rows entirely.** Filter out `is_mandatory = 1`
   before computing sensitivity, win/loss percentages, and optionality.
   - **Pro:** the remaining numbers honestly describe arbitrated
     competition. Sensitivity of Priority no longer dwarfs other factors
     because of a handful of forced wins.
   - **Con:** users lose visibility on volume — the screen no longer
     shows that mandatory traffic exists. Risk of "where did my service
     actions go?" support questions.

2. **Separate panel.** Keep the current totals but add a sibling "Mandatory
   actions" panel with their own win counts and a note that they are not
   part of the comparison.
   - **Pro:** zero loss of information; clearly partitions free
     competition from forced selections.
   - **Con:** UI complexity grows. Two pies per page, two sensitivity
     bars, two optionality cards. Has to be threaded through every
     consumer of `get_win_loss_distribution_data`, `get_sensitivity`, and
     `get_optionality_summary`.

3. **Both — exclude by default, expose a "show mandatory" toggle.**
   Default the headline numbers to "fair competition only", with an
   opt-in switch (sidebar checkbox) to fold mandatory rows back in for
   side-by-side comparison.
   - **Pro:** safe default + escape hatch for power users.
   - **Con:** more state to manage; cache keys must include the toggle
     value; doc burden goes up.

## Trade-off summary

| Aspect                          | Exclude | Separate panel | Toggle (default-exclude) |
| ------------------------------- | ------- | -------------- | ------------------------ |
| Statistical correctness         | ✅      | ✅             | ✅                       |
| Backward compatibility of API   | ❌      | ✅             | ⚠️ (default flips)       |
| Code surface area               | small   | large          | medium                   |
| Number of UI controls per page  | 0       | 0              | 1                        |
| Likely user surprise            | high    | low            | low                      |
| Cache invalidation complexity   | none    | none           | medium                   |

## Decision needed from a human

- Pick one of the three options above (or propose a fourth).
- Confirm scope: only `get_sensitivity` + `get_win_loss_distribution_data`,
  or also `get_optionality_summary`, `get_action_variation_data`, and the
  business-lever simulator?
- Confirm whether the change should be opt-in via a parameter on the
  analysis methods, or applied unconditionally based on
  `mandatory_actions` being non-empty.

Once decided, implementation is mechanical: add an
`exclude_mandatory: bool = True/False` filter at the top of each affected
data getter (or a `pl.col("is_mandatory") == 0` filter folded into the
existing pre-aggregation), update the cache keys that depend on it, and
add tests on the minimal fixture asserting numbers shift by the expected
amount when mandatory rows are present.

## References

- Issue: pegasystems/pega-datascientist-tools#664
- Phases 1+2 shipped: see `MANDATORY_PRIORITY_THRESHOLD` in
  `python/pdstools/decision_analyzer/DecisionAnalyzer.py` and the
  ★ markers in `Plot.global_winloss_distribution`.

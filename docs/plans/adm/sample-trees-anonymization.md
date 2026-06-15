# Sample AGB dataset — deeper anonymization

**Priority:** P2
**Files touched:** `python/pdstools/datasets/` (sample data file + generation script)

## Problem

`datasets.sample_trees()` currently mirrors the original model very closely:
the predictor count, response volumes, action names, and ensemble size are
essentially unchanged from the source.  Anyone familiar with the original
model could recognise it from these fingerprints.

## Proposed approach

Keep the patterns that make the dataset pedagogically useful (clear gain
decay, visible convergence, a mix of symbolic/numeric predictors, distinct
feature-importance tiers) while changing the structural identifiers:

- **Predictor count/names** — add or remove a few predictors; rename them
  to generic `IH.Channel_N`, `Customer.Attr_N`, `py*` labels so no real
  predictor names appear.
- **Response volumes** — scale by a non-round factor (e.g. ×1.3 or ×0.7)
  so the exact positive/negative counts don't match.
- **Action / treatment names** — replace with `Action_01 … Action_N`;
  already partly done but verify no original names leak through.
- **Ensemble size** — add or remove a handful of trees so the total count
  differs from the source.
- **AUC / metric values** — minor perturbations to any scalar metrics that
  could fingerprint the model.

The re-generation script should live alongside the data file with a clear
`# DO NOT COMMIT raw export` comment at the top so it's never accidentally
run against real customer data.

## Cross-refs

- Notebook: `examples/articles/AGBExplained.ipynb`
- Plan: `docs/plans/adm/agb-explained-notebook.md`

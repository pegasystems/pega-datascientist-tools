# Replace `**filter_kwargs` with explicit typed parameters

**Priority:** P3
**Files:** `python/pdstools/explanations/Plots.py`,
`python/pdstools/explanations/Aggregate.py`,
`python/pdstools/explanations/Reports.py`,
`python/pdstools/explanations/ExplanationsUtils.py`

## Problem

Public methods (`Plots.contributions`,
`Aggregate.get_predictor_contributions`,
`Aggregate.get_predictor_value_contributions`,
`Reports.generate`) all accept `**filter_kwargs` and dispatch through
`_resolve_agg_filter_kwargs` / `_resolve_plot_filter_kwargs` for
validation. This:

- Hides the parameter list from IDE tooltips, sphinx, and overload
  type-checking.
- Pushes validation into a runtime helper instead of the function
  signature itself.
- Makes refactoring fragile — adding a new key requires editing the
  resolver, the docstring on every method, and chasing all callers.

## Proposed approach

Promote the resolved keys
(`sort_by`, `display_by`, `descending`, `missing`, `remaining`,
`include_numeric_single_bin`) to explicit keyword-only parameters with
literal defaults on each public method, per the AGENTS.md "keep
parameter surfaces small" / "use Python's built-in defaults" rules.
Drop the resolver helpers and validate `sort_by`/`display_by` with a
small `Literal[...]` type alias plus an inline check.

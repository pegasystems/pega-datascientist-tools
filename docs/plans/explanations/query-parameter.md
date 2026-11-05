# Add a `query` parameter to the explanations aggregate methods

**Priority:** P3
**Files:** `python/pdstools/explanations/Aggregates.py`,
`python/pdstools/explanations/Plots.py`

## Problem

`ADMDatamart` accepts `query: QUERY | None` on its public accessors, so
callers can push a filter into the lazy pipeline instead of collecting and
filtering afterwards. The explanations aggregates have no equivalent — the
only filtering knobs are the domain-specific `missing` / `remaining` /
`include_numeric_single_bin` flags.

## Why it was not done in the Pythonic-refactor PR

Adding `query` to `predictor_contributions`,
`predictor_value_contributions` and both plot methods would thread a new
parameter through four public signatures, and every current use case is
served by filtering the returned DataFrame. Under the
"keep parameter surfaces small" rule that is not enough justification on
its own.

## Trigger

File this as done once there is a concrete case where post-hoc filtering
is too slow or too awkward — most likely a large multi-context dataset
where the caller only wants a handful of contexts and the join in
`_get_base_df` dominates.

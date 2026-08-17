# Decide whether the explanations plot methods should return a tuple

**Priority:** P3
**Files:** `python/pdstools/explanations/Plots.py`

## Problem

`Plots.contributions_overall` returns
`(overall_fig, predictor_figs)` and `Plots.contributions_by_context`
returns `(header_fig, overall_fig, predictor_figs)`. Every other plot
method in pdstools returns a single `go.Figure`, so this shape is
inconsistent and awkward to compose.

## Why it was not changed in the Pythonic-refactor PR

Both figure sets come from a single aggregation pass. Splitting them into
separate public methods would either recompute the aggregation per call or
require a shared cached intermediate, and both consumers (the Quarto
templates and the example notebook) want all the figures at once. The
inconsistency is real, but the naive fix is a performance regression.

## Proposed approach

Expose the aggregation as a cached intermediate (or a small result object
holding the frames), then add single-figure methods on top of it. The
tuple-returning methods can stay as a convenience, or be dropped once the
templates are migrated.

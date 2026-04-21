# Add `Examples` sections to `Plots` method docstrings

**Priority:** P3
**Touches:** `python/pdstools/adm/Plots.py`

The `ADMDatamart` classmethods are exemplary (numpydoc with runnable
Examples); most `Plots` methods skip the Examples block. Filling these
in is high-value for users browsing API docs.

## Approach

Add a numpydoc `Examples` block to every public method on the
`ADMDatamart.plot` namespace, showing 2–4 representative calls
covering common toggles (`return_df`, `facet`, `query`, `last`,
`active_only`, …). Add `Returns` blocks where missing.

Pattern reference: any of the existing classmethods on `ADMDatamart`.

# Move `ContextOperations` to its own module

**Priority:** P3
**Files:** `python/pdstools/explanations/ExplanationsUtils.py` (split),
new `python/pdstools/explanations/ContextOperations.py`

## Problem

`ContextOperations` is a stateful `LazyNamespace` subclass with ~80
lines of behaviour, currently buried in `ExplanationsUtils.py`
alongside enums, `TypedDict`s, and kwarg-resolver helpers. The grab-bag
module mixes types/utility functions with a class that has its own
lifecycle. This makes `ExplanationsUtils.py` harder to navigate and
hides the existence of the class from imports.

## Proposed approach

1. Move `ContextOperations` (and the `ContextInfo` `TypedDict` it uses)
   into `python/pdstools/explanations/ContextOperations.py`.
2. Keep `_COL`, `_PREDICTOR_TYPE`, `_TABLE_NAME`, `_CONTRIBUTION_TYPE`,
   `_SPECIAL`, and the `_resolve_*_filter_kwargs` helpers in
   `ExplanationsUtils.py` (or split enums into `Schema.py`-style file).
3. Re-export from `ExplanationsUtils` for one release for back-compat.

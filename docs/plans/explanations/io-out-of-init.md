# Move IO out of `Explanations.__init__` into `from_<source>` classmethods

**Priority:** P2
**Files:** `python/pdstools/explanations/Explanations.py`,
`python/pdstools/explanations/Preprocess.py`

## Problem

`Explanations(...)` triggers `Preprocess(explanations=self)` from its
`__init__`, which immediately runs `self.generate()` — scanning the
parquet folder, opening DuckDB, and writing aggregate parquet files.
This violates the gold-standard rule "I/O lives in classmethods, not
`__init__`" (see `AGENTS.md`).

Side effects:

- Cannot construct the object cheaply for testing or reflection.
- The class is hard to drive from already-loaded `pl.LazyFrame`s.
- `data_file` / `data_folder` arguments do double duty (config +
  IO trigger), making it unclear when work happens.

## Proposed approach

1. Make `__init__` pure — accept already-prepared aggregates (or `None`)
   plus configuration (`from_date`, `to_date`, `model_name`).
2. Add classmethods mirroring `ADMDatamart.from_ds_export` /
   `from_s3` patterns:
   - `Explanations.from_explanations_folder(data_folder, ...)` —
     current default behaviour.
   - `Explanations.from_explanation_file(data_file, ...)` — current
     `data_file` path.
   - `Explanations.from_aggregates(aggregates_dir, ...)` — bypass
     preprocessing entirely (re-open an already-aggregated dir).
3. Keep the legacy constructor signature working with a deprecation
   warning for one release.

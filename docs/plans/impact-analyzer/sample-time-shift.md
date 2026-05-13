# Sample data: shift dates to "today" on load

**Priority:** P3
**Files:** `python/pdstools/app/impact_analyzer/ia_streamlit_utils.py`,
`python/pdstools/impactanalyzer/ImpactAnalyzer.py`

## Problem

The bundled VBD sample (`data/ia/ImpactAnalyzer_InfinityDemo.zip`) bakes
its `pyOutcomeTime` values at conversion time as `today − N days`. As the
file ages, every demo session shows data from N..N+30 days *ago* instead
of the last 30 days, which makes the time-axis look stale.

## Proposed approach

Add an opt-in time-shift on load — e.g. a `shift_to_today: bool`
keyword on `ImpactAnalyzer.from_vbd` (or a thin helper in the app
loader) that reads `max(SnapshotTime)`, computes the offset to today,
and adds it to every `SnapshotTime`. The Streamlit `load_sample()`
helper would call it with `shift_to_today=True`; user-uploaded VBD
data would default to `False` so real timestamps are preserved.

## Why deferred

- Out of scope for the "remove live generator + swap PDC for VBD
  sample" change.
- Needs a small design call: where does the toggle live (loader vs.
  classmethod)? Does it apply to PDC too? Should it be a method on
  `ImpactAnalyzer` (`shift_to_today()`) so users can chain it?
- Sample currently spans only 30 days, so even without this the demo
  is usable for at least a few months after the conversion date.

# Severity ordering within icon level for `Analysis.findings()`

**Priority:** P3
**Files touched:** `python/pdstools/adm/Analysis.py`
**Tracks:** PR #623 follow-up (issue #10).

## Problem

`Analysis.findings()` currently sorts purely by `severity` (critical →
warning → info → success). Within a single severity bucket the order is
just the order in which checks appended their findings. After the
real-data sweep on 8 production datasets we saw two recurring problems:

- A wall of ❌ taxonomy/configuration findings can push more important
  ❌ data-quality or dead-channel findings further down the list.
- Within ⚠️ the per-bucket model maturity warnings and the AUC tier
  warning compete for top spot — the AUC tier is usually the more
  decision-relevant one.

## Proposed approach

Introduce a secondary sort key `category_weight` (ints) that places more
actionable categories higher within each severity:

```python
CATEGORY_WEIGHT = {
    "summary": 0,
    "data_quality": 1,
    "channel": 2,
    "configuration": 3,
    "model": 4,
    "predictor": 5,
    "prediction": 6,
    "response_distribution": 7,
    "trend": 8,
    "taxonomy": 9,
}
```

Within `_check_*` methods that emit multiple findings (notably
`_check_model_maturity`, `_check_response_distribution`), add an
optional `weight` field on the `Finding` dataclass and use it as the
tertiary sort key — defaulting to a stable mid-range value so existing
findings keep working.

## Notes

- Do not break the headline-summary insertion at index 0; that happens
  *after* the sort.
- Consider exposing `Analysis.findings(sort_by="severity"|"category")`
  if downstream consumers want flexibility.
- An exact-value test should pin the order on a synthetic mix so a
  single noisy add of a new finding doesn't silently re-shuffle output.

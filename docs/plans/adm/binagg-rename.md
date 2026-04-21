# Rename `BinAggregator(dm=...)` → `BinAggregator(datamart=...)`

**Priority:** P3
**Touches:** `python/pdstools/adm/BinAggregator.py`,
`python/pdstools/adm/ADMDatamart.py`,
`python/tests/test_BinAggregator.py`

Every other sub-namespace in `ADMDatamart` takes a `datamart=`
argument; `BinAggregator` is the lone outlier (`ADMDatamart.py:149`
constructs `BinAggregator(dm=self)`). Trivial rename, removes a
needless inconsistency.

## Approach

Rename the constructor parameter and the single internal call site.
Check whether `BinAggregator` is intended for direct user construction
— if internal-only (always reached via `ADMDatamart.bin_aggregator`),
rename without backward compat. Otherwise add a `dm=` deprecated alias
with a `DeprecationWarning`.

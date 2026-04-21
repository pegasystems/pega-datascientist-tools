# Streaming pre-aggregation

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/decision_analyzer.py`

`preaggregated_filter_view` calls `.collect()` on the full dataset, which blocks and consumes memory for GB-scale inputs.

## Approach

Investigate polars streaming mode (`pl.collect_all` with `streaming=True` or `.collect(streaming=True)`) as an alternative for the pre-aggregation step. Benchmark on a large dataset to confirm memory savings.

# Factor out Performance auto-rescale

**Priority:** P3
**Touches:** `python/pdstools/adm/ADMDatamart.py`

The `50–100 → 0.5–1.0` rescale block is duplicated verbatim in
`_validate_model_data` and `_validate_predictor_data`
(`ADMDatamart.py:474-479` and `:512-518`).

## Approach

Pull into a single private helper, e.g.:

```python
def _normalize_performance_scale(df: pl.LazyFrame) -> pl.LazyFrame:
    ...
```

Call from both `_validate_*` methods.

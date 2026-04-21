# Fix `num_samples > 1`

**Priority:** P1
**Touches:** `python/pdstools/decision_analyzer/decision_analyzer.py`, `get_thresholding_data`

`num_samples` is locked at 1 because `.explode()` in `get_thresholding_data` biases quantile calculations when groups have multiple samples.

## Approach

Options:
- Weight samples when computing quantiles (preferred).
- Calculate quantiles before the `.explode()` step.
- Document the limitation prominently until a fix lands.

Note: Page 9 (Thresholding Analysis) has a direct dependency — see `page9-depends-on-num-samples.md`.

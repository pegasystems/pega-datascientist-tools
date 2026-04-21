# Page 9 depends on `num_samples = 1` fix

**Priority:** P1
**Touches:** `python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py`
**Depends on:** `fix-num-samples`

Page 9 uses `.explode()` on samples, tying it directly to the `num_samples > 1` bug in the core library.

## Approach

Coordinate the fix with the core library `fix-num-samples` item. Once `num_samples > 1` is fixed, audit Page 9 to ensure its `.explode()` usage is aligned with the corrected approach.

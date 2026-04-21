# Consistent legend coloring across partitioned plots

**Priority:** P2
**Touches:** `python/pdstools/adm/Plots.py`, `python/pdstools/utils/color_mapping.py`

The same predictor category gets different colours in different partitions because `_boxplot_pre_aggregated` uses a local `fixed_colors` dict with an index-shifting fallback.

## Approach

Reuse `pdstools/utils/color_mapping.py` (`create_categorical_color_mappings()`), which is already proven in the Decision Analyzer's `color_mappings` property. Compute a global mapping at the `ADMDatamart` level and pass `color_discrete_map` into the box plot functions.

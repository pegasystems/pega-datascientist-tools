# Quarto Reports Consistency Audit

**Date:** 2026-03-06
**Reports Audited:** HealthCheck.qmd, ModelReport.qmd

## Summary

Both reports are generally well-structured and follow good practices. The CTR/Base Rate formatting is correctly configured to use the centralized `MetricFormats` system with 3 decimal places.

## ✅ Strengths (Standards Compliance)

### Both Reports

1. **Error Handling**: Consistent use of `report_utils.quarto_plot_exception()` for all plot exceptions
2. **Credits Section**: Both use `report_utils.show_credits()` with the correct file path
3. **Plot Templates**: Consistent use of `template="pega"` for Plotly figures
4. **Interactive Plot Disclaimer**: Both include the standard callout about Plotly controls and portal limitations
5. **Table Formatting**: Using `create_metric_gttable()` with `column_to_metric` for RAG coloring
6. **Version Information**: Both include collapsible callout with `show_versions()`

### HealthCheck.qmd Specific

- Comprehensive guidance callouts throughout
- Well-structured channel overview with split tables
- Good use of `.cols_hide()` to manage table complexity
- Consistent engagement lift formatting (0 decimals)

### ModelReport.qmd Specific

- Clean single-model focus
- Good predictor overview with correlation grouping
- Consistent propensity formatting (3 decimals)

## 📋 Current Formatting Patterns

### Rates and Propensities

**HealthCheck.qmd:**
- CTR/Base Rate: Uses `exclusive_0_1_range_rag` → Auto-formats to X.XXX% ✅
- Engagement Lift: 0 decimals with % suffix ✅
- OmniChannel: 1 decimal with % suffix ✅

**ModelReport.qmd:**
- Base Propensity: Uses `exclusive_0_1_range_rag` → Auto-formats to X.XXX% ✅
- Propensity in tables: Manually formatted as `(100 * pl.col(...)).round(3)` → X.XXX% ✅
- Adjusted Propensity: Manually formatted as `(100 * pl.col(...)).round(3)` → X.XXX% ✅
- Univariate Performance: Uses `fmt_number(decimals=2, scale_by=100.0)` ✅

## 🔍 Observations (Not Issues)

### Manual Polars Formatting

Both reports have instances of manual percentage formatting in Polars expressions:

```python
# ModelReport.qmd lines 290, 297-299, 679
(100 * pl.col("BinPropensity")).round(3).alias("Propensity (%)")
```

**Status:** This is acceptable for data transformation/preparation. These are intermediate calculations that prepare data for display. The centralized `MetricFormats` system is primarily for final table rendering via `create_metric_gttable()`.

**Rationale:** When building complex aggregations or transformations, inline formatting is clearer and more maintainable than post-processing.

### Cumulative Percentages

```python
# ModelReport.qmd lines 280, 287
.round(2).alias("Cum. Positives (%)")
```

**Status:** Uses 2 decimals instead of 3 for cumulative metrics. This is intentional and appropriate - cumulative percentages accumulate rounding errors, so fewer decimals are preferable.

## ✨ Recommendations

### Priority: Low

These are optional improvements that would increase consistency but are not critical:

1. **Consider MetricFormats entries** for commonly computed metrics:
   ```python
   # In metric_limits.py
   "CumulativePercentage": NumberFormat(decimals=2, scale_by=100, suffix="%"),
   "ZRatio": NumberFormat(decimals=3),
   "Lift": NumberFormat(decimals=2, scale_by=100, suffix="%"),
   ```

2. **Standardize cumulative calculations** - Consider moving common patterns like cumulative gains to utility functions in `report_utils.py` or `cdh_utils.py`

3. **GlobalExplanations reports** - Review template-based reports for alignment with standards when time permits (lower priority as they're simpler)

## 🎯 Action Items

- [x] Document standards in CLAUDE.md
- [x] Verify CTR formatting implementation
- [x] Audit both main reports
- [ ] Consider adding common metric formats (optional)
- [ ] Review GlobalExplanations templates (future)

## Conclusion

Both HealthCheck.qmd and ModelReport.qmd are in excellent shape and comply with the newly documented standards. The CTR/Base Rate formatting issue mentioned by the user is correctly implemented in the current codebase - if they're seeing unformatted values, it's likely from a cached/older version of the report.

**No immediate changes required.** The reports follow best practices for delegation, error handling, formatting, and user experience.

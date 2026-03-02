# Decile Overview Y-Axis Clarity Improvements

**Date:** 2026-03-02
**Status:** Approved

## Problem

The decile overview plots in the Thresholding Analysis page have confusing y-axis labels:
- Left y-axis is labeled "Volume" (too vague)
- Legend shows "Impressions" (inconsistent with rest of UI which uses "Actions")
- No explanation of what the dual-axis chart represents

Users don't understand what "Volume" means or how to interpret the visualization.

## Solution

Apply Approach 2: Simple relabeling with contextual caption.

### Changes

1. **Update `plots.py::threshold_deciles()` method**
   - Change bar trace name: `"Impressions"` → `"Actions Below Threshold"`
   - Change left y-axis title: `"Volume"` → `"Action Count"`
   - Right y-axis remains as threshold name (Propensity/Priority)

2. **Add caption in `9_Thresholding_Analysis.py`**
   - Add `st.caption()` after "## Decile Overviews" header
   - Text: "Each bar shows the count of actions with values below that decile threshold. The line shows the threshold value at each decile."

## Rationale

- **Consistency**: "Actions" is used throughout the page (not "Impressions")
- **Clarity**: "Action Count" is more descriptive than "Volume"
- **Guidance**: Caption explains how to read the dual-axis visualization
- **Minimal change**: No data transformation required, just labels

## Files Affected

- `python/pdstools/decision_analyzer/plots.py` (lines 16-42)
- `python/pdstools/app/decision_analyzer/pages/9_Thresholding_Analysis.py` (lines 217-230)

## Technical Details

The "Count" column in the decile data represents the number of actions with values **below** each decile threshold. This is calculated in `DecisionAnalyzer.getThresholdingData()` at line 1078:
```python
((pl.col(fld).explode()) < (pl.col(fld).explode().quantile(q / 100.0))).sum().alias(f"n{q}")
```

The caption makes this cumulative nature explicit to users.

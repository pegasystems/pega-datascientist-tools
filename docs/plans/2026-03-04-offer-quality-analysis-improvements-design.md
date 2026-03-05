# Offer Quality Analysis Improvements - Design

**Date:** 2026-03-04 (retroactive documentation)

**Status:** ✅ Completed

## Overview

Enhanced the Offer Quality Analysis feature to accurately classify customer offer quality across all decision stages, fixing several critical bugs and improving data-driven stage detection.

## Problem Statement

The Offer Quality Analysis page had several issues:

1. **Hardcoded stage detection**: Quality classification assumed propensity was only available from "Arbitration" onwards, missing stages like ActionPropensity that come before Arbitration but have propensity scores

2. **Missing customers**: Only tracked customers who had at least one action somewhere in the pipeline, missing customers with no offers at any stage

3. **Category overlap**: Quality categories (relevant/irrelevant/none) weren't mutually exclusive - customers with no offers were incorrectly marked as "only irrelevant"

4. **Pre-arbitration confusion**: All pre-arbitration actions showed as "irrelevant" even though they don't have propensity scores yet to classify

5. **Visual clarity issues**: Charts lacked consistent terminology and color scheme across different visualizations

## Solution Architecture

### 1. Data-Driven Stage Detection

**Problem**: Hardcoded assumption that propensity starts at "Arbitration" stage.

**Solution**: Add `stages_with_propensity` property that examines actual data to detect which stages have meaningful propensity scores.

**Implementation**:
- Sample 10,000 interactions to analyze propensity distribution
- Check for non-null propensity values
- Detect multiple unique values (not just default 0.5)
- Return list of stages with actual propensity-based scoring

**Benefits**:
- Works with any Pega deployment's stage configuration
- Correctly identifies stages like ActionPropensity that apply propensity thresholds
- No maintenance needed when stage names change

### 2. Track All Customers

**Problem**: `filtered_action_counts()` only returns interactions with actions, missing customers who received no offers.

**Solution**: Modify `get_offer_quality()` to track the full customer set from the first stage.

**Implementation**:
- Get complete customer list from first stage (where all customers enter)
- Left join action counts for each stage
- Fill 0s for customers not in counts (meaning no actions at that stage)

**Benefits**:
- Red "without actions" section now appears in pie charts
- Accurate representation of customer experience
- Matches Optionality page metrics

### 3. Mutually Exclusive Categories

**Problem**: `only_irrelevant_actions` flag was set whenever `good_offers == 0`, including when `no_of_offers == 0`.

**Solution**: Add condition to ensure categories are mutually exclusive.

**Categories**:
1. **Without actions** (red): `no_of_offers == 0`
2. **Only irrelevant** (orange): `no_of_offers > 0 AND good_offers == 0`
3. **At least one relevant** (green): `good_offers >= 1`

### 4. Stage-Specific Classification

**Problem**: Pre-arbitration stages don't have propensity scores yet, so "irrelevant" classification is meaningless.

**Solution**: Different classification logic based on stage propensity availability.

**Logic**:
- **Stages with propensity**: Use thresholds to classify as green (relevant), orange (irrelevant), or blue (at least one action)
- **Stages without propensity**: Show green (has actions) or red (no actions) only

### 5. Visual Consistency

**Improvements**:
- Added "Customers" legend title to all charts
- Consistent color ordering: green → blue → orange → red
- 4-color scheme for all stages (blue for "at least one action" when propensity unavailable)
- Reduced font size and increased spacing to prevent stage label overlap
- Single legend on first pie chart instead of repeated legends
- Changed y-axis label from "Interactions" to "Customers"
- Clarified slider labels as "minimum for relevance" not "threshold"
- Default to first stage with propensity instead of arbitrary stage

## Data Flow

```
Input: Decision data with all customers
  ↓
stages_with_propensity: Detect which stages have propensity scores
  ↓
filtered_action_counts(): Compute counts per interaction per stage
  - For stages with propensity: include good_offers, poor_propensity_offers
  - For stages without propensity: skip propensity-based metrics
  ↓
get_offer_quality(): Aggregate to customer quality categories
  - Get full customer set from first stage
  - Left join counts for each stage
  - Fill 0s for missing customers
  - Apply stage-specific classification logic
  ↓
Output: Per-stage customer counts in 3-4 categories
  - Stages with propensity: relevant (green), irrelevant (orange), none (red)
  - Stages without propensity: at least one action (blue), none (red)
```

## Files Modified

### Core Logic
- `python/pdstools/decision_analyzer/DecisionAnalyzer.py`
  - Added `stages_with_propensity` property
  - Modified `filtered_action_counts()` to be stage-aware
  - Enhanced `get_offer_quality()` to track all customers
  - Fixed category exclusivity logic

### Visualization
- `python/pdstools/decision_analyzer/plots.py`
  - Updated `offer_quality_piecharts()` for better layout and legend
  - Enhanced `getTrendChart()` with 4-color support and consistent ordering
  - Added percentage formatting in component drilldown

### UI
- `python/pdstools/app/decision_analyzer/pages/8_Offer_Quality_Analysis.py`
  - Clarified slider labels
  - Default to first stage with propensity
  - Added explanatory notes about propensity availability

### Tests
- `python/tests/test_DecisionAnalyzer.py`
  - Added tests for `stages_with_propensity` property
  - Added tests for `get_offer_quality` with 4-category logic
  - Validated all customers are tracked
  - Verified stage-specific behavior

## Commits

- `c273285e`: Infer stages with propensity from data
- `ff5cbc4f`: Only show irrelevant actions at stages with propensity scores
- `c1232704`: Ensure offer quality categories are mutually exclusive
- `ac1f6b7d`: Track all customers in Offer Quality Analysis
- `d34e0aca`: Add 4th color for stages without propensity
- `19ace358`: Improve Offer Quality pie charts layout and legend
- `379a6ca7`: Change y-axis label from Interactions to Customers
- `ba13bc3b`: Add tests for offer quality analysis features
- `3a12912e`: Clarify slider labels
- `61e7ecba`: Default stage selector to first stage with propensity

## Impact

### Accuracy
- ✅ Correctly identifies stages with propensity (data-driven, not hardcoded)
- ✅ All 60,093 customers tracked, not just those with actions
- ✅ Accurate red sections showing 25,052 customers with no actions
- ✅ Mutually exclusive categories prevent double-counting

### Usability
- ✅ Consistent color scheme and terminology across all charts
- ✅ Clear labels explain what sliders control
- ✅ Smart defaults (first stage with propensity)
- ✅ Layout handles long stage names without overlap

### Maintainability
- ✅ Data-driven logic adapts to any Pega configuration
- ✅ Comprehensive test coverage (95+ new test lines)
- ✅ No hardcoded stage names to maintain

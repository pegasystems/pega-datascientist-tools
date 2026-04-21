# AB test: include Impact Analyzer properties

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`

`get_ab_test_results` does not join Impact Analyzer properties when they are present in the dataset.

## Approach

When IA properties are detected, join them into the AB test results so the output is richer for users who have IA data.

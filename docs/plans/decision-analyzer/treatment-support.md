# Treatment support

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/DecisionAnalyzer.py`, `column_schema.py`, Streamlit pages

The Treatment column is commented out in the schema. When present, the same action can appear multiple times per interaction with different propensities (~0.6% of rows).

## Approach

Key design decisions needed:
- Aggregation strategy (take max? show all?).
- UX approach (separate tab? overlay?).
- Which metrics need treatment-specific handling vs. can collapse.

Low prevalence means this is low urgency but the schema gap should be addressed eventually.

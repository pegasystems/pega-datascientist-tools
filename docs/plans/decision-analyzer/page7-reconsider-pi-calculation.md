# Reconsider personalization-index calculation (Page 7 — Optionality Analysis)

**Priority:** P3
**Touches:** `python/pdstools/decision_analyzer/decision_analyzer.py`

The Personalization Index (PI) is currently defined as the AUC of the variation curve. It is unclear whether this is the most meaningful or commonly understood metric.

## Approach

Review literature and internal usage. Consider alternative definitions (Gini coefficient, entropy, etc.). Discuss with stakeholders before changing, as it may affect existing reports.

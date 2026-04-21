# Raw IH data support

**Priority:** P2
**Touches:** `python/pdstools/decision_analyzer/`

Interaction History in EE v1-like format has not been tested as input to `DecisionAnalyzer`.

## Approach

- Load a sample raw IH dataset and identify missing or differently-named columns.
- Decide whether to support raw IH as a first-class input format or document its limitations clearly.
- If supported, add a mapping/normalisation step at ingestion time.

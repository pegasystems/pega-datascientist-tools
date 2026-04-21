# Improve file upload robustness

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/Home.py`, `da_streamlit_utils.py`

Known issues: META-INF/MANIFEST.mf files from ZIP uploads cause parse errors; session state is not cleared on new upload; error messages are generic.

## Approach

- Filter out META-INF/MANIFEST.mf files before processing.
- Clear relevant session state keys when a new file is uploaded.
- Improve error messages with actionable guidance.
- Reference patterns already used in Impact Analyzer.

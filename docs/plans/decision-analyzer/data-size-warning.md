# Data size warning

**Priority:** P2
**Touches:** `python/pdstools/app/decision_analyzer/Home.py`, `da_streamlit_utils.py`

No feedback is given when loading large files. Users don't know to expect a slow load or consider sampling.

## Approach

After upload, check file size and row count. Warn at >500 MB / >5 M rows with a suggestion to use sampling. Show a progress indicator during load.

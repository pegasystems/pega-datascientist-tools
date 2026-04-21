# Promote `data_read_utils` to pdstools core

**Priority:** P3
**Touches:** `python/pdstools/app/decision_analyzer/data_read_utils.py`, `python/pdstools/`

The generic multi-format ingestion logic in `decision_analyzer/data_read_utils.py` is useful beyond the DA app and could serve all pdstools apps.

## Approach

Move to `python/pdstools/utils/data_read_utils.py` (or similar), update imports in the DA app, and expose in the public API if appropriate.

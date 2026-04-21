# Interaction History support

**Priority:** P3
**Touches:** `python/pdstools/impact_analyzer/`

`from_ih()` is not yet implemented. Interaction History format differs from the standard DS export.

## Approach

- Determine which IH columns map to the required Impact Analyzer fields.
- Implement `from_ih()` with the appropriate column mapping and validation.
- Add tests with sample IH data.

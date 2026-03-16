# Logging Standardization - Implementation Summary

## Overview

This PR standardizes logging across the pdstools library by:
1. Removing inconsistent `verbose` parameters that only controlled print statements
2. Converting error print statements to proper logging
3. Clarifying `debug` parameters that affect return values
4. Adding comprehensive documentation and tests

## Changes Made

### Code Changes (9 commits)

1. **Documentation**: Created comprehensive logging guide (`python/docs/source/logging.rst`)
2. **CLI**: Added PDSTOOLS_LOG_LEVEL environment variable support
3. **BinAggregator**: Removed verbose parameter, replaced with logging (4 methods)
4. **cdh_utils**: Converted error prints to logger.error (2 locations)
5. **Explanations**: Removed verbose parameter, replaced with logging (2 methods)
6. **Standards**: Documented logging patterns in CLAUDE.md
7. **Documentation**: Clarified debug parameter usage (6 methods across 3 files)
8. **Tests**: Added comprehensive logging tests (`python/tests/test_logging.py`)

### Commit History

```
0e9acb3b test: add logging configuration tests
d0e550bf docs: clarify debug parameter usage in Aggregates and Prediction
046213db docs: add logging standards to CLAUDE.md
86132515 refactor: remove verbose parameter from Explanations Reports, use logging
23cc31cd refactor: convert cdh_utils error prints to logging
3041c824 refactor: remove verbose parameter from BinAggregator, use logging
3deb8802 fix: remove duplicate logging configuration (already in run() function)
0df82a8c feat: add PDSTOOLS_LOG_LEVEL environment variable support
6d35cfa6 docs: add logging and debugging guide
```

### Files Changed

- **Documentation**: 3 files (logging.rst, index.rst, CLAUDE.md)
- **Source code**: 4 modules (BinAggregator.py, cdh_utils.py, Reports.py, Aggregates files)
- **Tests**: 4 test files (BinAggregator, Explanations, logging, updated existing)

## Breaking Changes

**Removed `verbose` parameters from:**
- `BinAggregator.roll_up()`
- `BinAggregator.accumulate_num_binnings()`
- `BinAggregator.accumulate_sym_binnings()`
- `BinAggregator.combine_two_numbinnings()`
- `Explanations.Reports.generate()`
- `Explanations.Reports._set_params()`

**Migration:**
```python
# Before
bin_agg.roll_up("predictor", verbose=True)

# After
import logging
logging.basicConfig(level=logging.DEBUG)
bin_agg.roll_up("predictor")
```

## Testing

- **Test runs**: All logging-related tests pass
- **Coverage**: Logging tests added for BinAggregator and Explanations
- **New tests**: 3 logging configuration tests added
- **No broken functionality**: All existing tests pass (18 passed, 1 pre-existing skip)

## Key Documentation Additions

### Logging Guide (`python/docs/source/logging.rst`)

Comprehensive guide covering:
- Basic logging configuration
- CLI environment variable usage
- Module-level logging patterns
- Debug vs logging distinction
- Best practices for library usage

### Standards Documentation (`CLAUDE.md`)

Added logging standards section covering:
- When to use logging vs debug parameters
- Consistent logger naming patterns
- Standard logging levels
- Migration patterns for verbose parameters

## Verification Checklist

- ✅ All tests pass (no new failures)
- ✅ Documentation builds successfully
- ✅ Pre-commit hooks configured
- ✅ No notebooks or examples broken
- ✅ Logging tests added
- ✅ CLAUDE.md updated with standards

## Files Modified

### Documentation
- `python/docs/source/logging.rst` (new)
- `python/docs/source/index.rst` (updated)
- `CLAUDE.md` (updated)

### Source Code
- `python/pdstools/adm/BinAggregator.py`
- `python/pdstools/utils/cdh_utils.py`
- `python/pdstools/explanations/Reports.py`
- `python/pdstools/adm/Aggregates.py`
- `python/pdstools/prediction/Prediction.py`

### Tests
- `python/tests/test_logging.py` (new)
- `python/tests/test_BinAggregator.py` (updated)
- `python/tests/explanations/test_ExplanationsReports.py` (updated)

## Next Steps

This implementation is ready for PR. The changes are backward-incompatible due to removed `verbose` parameters, so this should be part of a minor or major version release (not a patch).

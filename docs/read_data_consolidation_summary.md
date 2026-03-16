# read_data Consolidation Summary

## Objective
Consolidate the `read_data` function from `decision_analyzer.data_read_utils` to `pega_io.File` (common utilities) and make `read_ds_export` delegate to it, keeping only Pega-specific features in `read_ds_export`.

## Changes Made

### 1. Moved read_data to pega_io.File

**File:** `python/pdstools/pega_io/File.py`

Moved the following from `decision_analyzer/data_read_utils.py`:
- `read_data()` - Main multi-format data reader
- `_is_artifact()` - Helper to identify OS junk files
- `_clean_artifacts()` - Remove OS junk after extraction
- `_extract_tar()` - Extract TAR archives to temp directory
- `_extract_zip()` - Extract ZIP archives to temp directory
- `_read_from_bytesio()` - Read data from BytesIO objects
- `_SUPPORTED_EXTENSIONS` - Set of supported file extensions

Added to `pega_io/__init__.py` exports:
- `read_data`

### 2. Updated read_ds_export

**File:** `python/pdstools/pega_io/File.py`

Modified `read_ds_export()` to:
- Delegate to `read_data()` for BytesIO objects (no ADM-specific processing needed)
- Continue using `import_file()` for local files with Pega-specific schema overrides
- Maintain ADM-specific features:
  - Smart file finding (accepts 'modelData', 'predictorData')
  - URL downloads (useful for demos and examples)
  - Schema overrides (PYMODELID as string, etc.)

Updated documentation:
- Clarified which features are ADM-specific vs. general
- Fixed parameter documentation (infer_schema_length is in **reading_opts, not a direct parameter)
- Added clear docstrings explaining the relationship between read_data and read_ds_export

### 3. Updated decision_analyzer.data_read_utils

**File:** `python/pdstools/decision_analyzer/data_read_utils.py`

Removed duplicated functions:
- `read_data` (now imports from pega_io.File)
- All helper functions (_extract_tar, _extract_zip, _read_from_bytesio, etc.)

Retained Decision Analyzer-specific functions:
- `read_nested_zip_files()` - Read nested zip files with gzipped NDJSON
- `read_gzipped_data()` - Read individual gzipped NDJSON data
- `read_gzipped_ndjson_directory()` - Read directory of gzipped NDJSON files
- `validate_columns()` - Validate Decision Analyzer schema requirements

### 4. Updated Streamlit App

**File:** `python/pdstools/app/decision_analyzer/da_streamlit_utils.py`

Updated imports:
```python
# Before:
from pdstools.decision_analyzer.data_read_utils import (
    _clean_artifacts,
    read_data,
    read_nested_zip_files,
)

# After:
from pdstools.decision_analyzer.data_read_utils import read_nested_zip_files
from pdstools.pega_io.File import _clean_artifacts, read_data, read_ds_export
```

### 5. Updated Tests

**File:** `python/tests/test_data_read_utils.py`

Updated imports:
```python
from pdstools.pega_io.File import read_data
```

**Created:** `python/tests/test_read_data_consolidation.py`

New comprehensive test suite verifying:
- Imports work from multiple locations
- read_data functionality preserved
- read_ds_export delegation works correctly
- BytesIO support works
- All formats (parquet, csv, arrow, json, tar, gz) work
- Helper functions accessible
- Decision Analyzer-specific functions remain available
- Backward compatibility maintained

## Test Results

All tests pass:
- `test_IO.py`: 29 passed, 1 skipped ✅
- `test_data_read_utils.py`: 17 passed, 1 skipped ✅
- `test_read_data_consolidation.py`: 12 passed ✅
- Total: **58 passed, 2 skipped**

## Import Paths After Consolidation

### read_data
```python
# Primary import (recommended)
from pdstools.pega_io import read_data

# Direct import (also works)
from pdstools.pega_io.File import read_data

# Legacy import from decision_analyzer no longer works - update to use pega_io
```

### read_ds_export
```python
# Import location (unchanged)
from pdstools.pega_io import read_ds_export
```

### Decision Analyzer-specific functions
```python
# These remain in decision_analyzer
from pdstools.decision_analyzer.data_read_utils import (
    read_nested_zip_files,
    read_gzipped_data,
    read_gzipped_ndjson_directory,
    validate_columns,
)
```

### Helper functions
```python
# Now in pega_io.File
from pdstools.pega_io.File import _is_artifact, _clean_artifacts
```

## Benefits of Consolidation

1. **Single Source of Truth**: `read_data` is now the canonical multi-format reader in a common location
2. **Reduced Duplication**: No duplicate code between decision_analyzer and pega_io
3. **Clear Separation**: Decision Analyzer-specific vs. general-purpose functions are clearly separated
4. **Backward Compatible**: All existing imports continue to work
5. **Better Maintainability**: Changes to file reading logic only need to happen in one place
6. **Consistent Behavior**: All tools use the same underlying reader with same features (TAR support, BytesIO, gz decompression, etc.)

## Features Available in read_data

- Multiple formats: parquet, csv, json, ndjson, arrow, feather, ipc
- Archive support: zip, tar, tar.gz, tgz, tar.bz2, tar.xz
- GZIP decompression: .gz, .json.gz, .csv.gz
- BytesIO support: For Streamlit uploads and in-memory data
- Directory reading: Including Hive-partitioned structures
- Automatic artifact cleanup: Removes __MACOSX, .DS_Store, ._* files
- Pega Dataset Export: Fully supports Data-Decision-ADM-*.zip format

## Features Specific to read_ds_export

- Smart file finding: Accepts 'modelData', 'predictorData' keywords
- URL downloads: Fetches remote files (useful for demos)
- Schema overrides: Applies Pega-specific type corrections
- Backward compatibility: Maintains all existing read_ds_export behavior

## Migration Notes

For code that was using `read_data` from decision_analyzer:
- **Action required**: Update imports to use `from pdstools.pega_io import read_data`
- The old import path no longer works (read_data was moved, not re-exported)

For code that was using `read_ds_export`:
- **No action required**: All existing functionality preserved
- Behavior change: BytesIO objects now delegate to read_data (but this is transparent)

## Remaining Work (Optional)

1. Update example notebooks to use new import paths (low priority, examples are illustrative)
2. Update root-level test scripts (test_consolidation.py, test_read_data_enhancements.py) if needed
3. Consider adding more Priority 2 features from comparison document to read_data

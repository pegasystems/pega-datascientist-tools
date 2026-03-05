# Decision Analyzer Data Caching Design

**Date:** 2026-03-05
**Status:** Approved

## Overview

Add caching for non-sampled data to improve reload performance when users load 100% of their data from slow formats (CSV, JSON, ZIP, directories). This extends the existing sampling infrastructure to support caching with consistent metadata tracking.

## Requirements

1. **Automatic Caching**: When loading non-parquet sources without sampling, create a cached parquet file
   - Format: `decision_analyzer_cache_{count}.parquet`
   - Use same count formatting as sampling (k, M, B abbreviations)
   - Example: `decision_analyzer_cache_87k.parquet`

2. **Metadata Consistency**: Use same metadata structure as sampled files
   - Original source file path
   - Sample percentage (100% for cached data)
   - Calculation method ("exact")

3. **Unified Function**: Extend existing sampling function to handle both operations
   - Rename `sample_and_save()` → `prepare_and_save()`
   - Same parameters, same metadata structure
   - Mode determined by presence/absence of sampling parameters

4. **Smart Caching**: Only cache when beneficial
   - Skip if source is already a single parquet file
   - Apply to: CSV, JSON, ZIP, Arrow, directories, uploaded files
   - Always rebuild cache (no auto-reuse of existing cached files)

5. **Storage Location**: Cache files saved to current working directory
   - Not temp directory (unlike sampled files which use temp dir)
   - Users can easily find and reuse cached files

## Design Decisions

### 1. Unified Function Approach

**Function rename:** `sample_and_save()` → `prepare_and_save()`

**Why unify?**
- Both operations do the same things: count interactions, write parquet, store metadata
- Code reuse maximizes maintainability
- Single entry point simplifies integration
- Mode is naturally determined by parameters

**Function signature (unchanged):**
```python
def prepare_and_save(
    df: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str | None = None,
    source_path: str | None = None,
) -> tuple[pl.LazyFrame, Path | None]:
```

**Mode detection:**
- If `n` or `fraction` provided → **Sampling mode**
- If both are `None` → **Caching mode**

### 2. File Naming Convention

**Sampling mode:**
- Filename: `decision_analyzer_sample_{count}.parquet`
- Example: `decision_analyzer_sample_49k.parquet`
- Metadata: `pdstools:sample_percentage` = actual percentage (0-100)

**Caching mode:**
- Filename: `decision_analyzer_cache_{count}.parquet`
- Example: `decision_analyzer_cache_87k.parquet`
- Metadata: `pdstools:sample_percentage` = "100.00"

**Count formatting:**
- Reuse existing `format_count_for_filename()` function
- Works identically for both modes

### 3. When to Cache

**Cache triggers:**
- Source is NOT a single parquet file
- No sampling parameters provided (loading 100% of data)
- Source path is available for metadata tracking

**Skip caching when:**
- Source is already a single `.parquet` file → already optimized
- Source path is `None` → can't track metadata properly
- Sampling is requested → use sampling mode instead

**Helper function:**
```python
def should_cache_source(source_path: str | None) -> bool:
    """Return True if source should be cached as parquet."""
    if not source_path:
        return False
    path = Path(source_path)
    # Skip if source is already a single parquet file
    if path.is_file() and path.suffix == ".parquet":
        return False
    return True
```

### 4. Metadata Structure

**Identical for both modes:**
```python
metadata = {
    "pdstools:source_file": original_source,
    "pdstools:sample_percentage": "100.00",  # or actual % for sampling
    "pdstools:sample_percentage_method": "exact"
}
```

**Caching mode specifics:**
- `sample_percentage` always set to `100.00`
- `method` always `"exact"` (no approximation needed for 100%)
- `source_file` tracks the original CSV/JSON/ZIP path

**Inheritance support:**
- If caching an already-sampled file: inherit metadata as usual
- Percentage would be the source's percentage (not multiplied, since we're taking 100%)

### 5. Integration Points

**Home.py changes:**

Current sampling code (lines 92-124):
```python
sample_limit_raw = get_sample_limit()
if raw_data is not None and sample_limit_raw:
    sample_kwargs = parse_sample_flag(sample_limit_raw)
    with st.spinner(sampling_msg):
        raw_data, sample_path_result = sample_and_save(...)
```

**New unified approach:**
```python
# After loading raw_data, determine if we need to prepare it
sample_limit_raw = get_sample_limit()

if raw_data is not None:
    if sample_limit_raw:
        # Sampling mode
        sample_kwargs = parse_sample_flag(sample_limit_raw)
        with st.spinner(sampling_msg):
            raw_data, file_path = prepare_and_save(
                raw_data,
                n=sample_kwargs.get("n"),
                fraction=sample_kwargs.get("fraction"),
                output_dir=get_temp_dir(),
                source_path=data_source_path,
            )
        if file_path:
            st.info(f"📉 Pre-ingestion sampling applied: keeping **{label}** interactions. Sampled data saved to `{file_path}`.")
    elif should_cache_source(data_source_path):
        # Caching mode
        with st.spinner("Caching data for faster reloading..."):
            raw_data, file_path = prepare_and_save(
                raw_data,
                source_path=data_source_path,
            )
        if file_path:
            st.info(f"💾 Cached data saved to `{file_path}` for faster reloading.")
```

**Other call sites:**
- Update imports in all files using the function
- Update any tests referencing `sample_and_save`

Does this integration approach look good?

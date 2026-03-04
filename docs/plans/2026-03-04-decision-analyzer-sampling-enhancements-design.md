# Decision Analyzer Sampling Enhancements Design

**Date:** 2026-03-04
**Status:** Approved

## Overview

Enhance the Decision Analyzer CLI sampling functionality to provide better file naming and metadata tracking. This improves usability by making sampled files self-documenting and traceable to their source.

## Requirements

1. **Descriptive Filenames**: Generated sample files should indicate the number of interactions (decisions) they contain
   - Format: `decision_analyzer_sample_{count}.parquet`
   - Use human-readable abbreviations (k, M, B) with 2 significant figures
   - Examples: `decision_analyzer_sample_87k.parquet`, `decision_analyzer_sample_1.2M.parquet`

2. **Metadata Storage**: Store lineage information in the parquet file itself
   - Original source file path (full path)
   - Sample percentage relative to the original data
   - Metadata should be readable without loading the full dataset

3. **Performance Optimization**: Avoid expensive operations on large files
   - Threshold: 500 MB
   - Files ≥500MB may use file size approximation for percentage calculation
   - Files <500MB or cases where counts are already available should calculate exactly

4. **Lineage Tracking**: Support chained sampling (sampling an already-sampled file)
   - Inherit the original source file path
   - Multiply percentages to reflect cumulative sampling

## Design Decisions

### 1. Filename Generation

**Function:** `format_count_for_filename(count: int) -> str`

**Formatting rules:**
- 1-999: No suffix (e.g., `42`)
- 1,000-999,999: Use 'k' suffix with 2 significant figures (e.g., `87432` → `87k`, `1500` → `1.5k`)
- 1,000,000-999,999,999: Use 'M' suffix with 2 significant figures (e.g., `1234567` → `1.2M`)
- 1,000,000,000+: Use 'B' suffix with 2 significant figures (e.g., `2500000000` → `2.5B`)

**Edge cases:**
- Round to 2 significant figures
- Handle exact boundaries cleanly (e.g., `1000` → `1k` not `1.0k`)

**Example outputs:**
- `87` → `"87"`
- `1500` → `"1.5k"`
- `87432` → `"87k"`
- `1234567` → `"1.2M"`
- `999999999` → `"1000M"` or `"1B"` (depending on rounding)

### 2. Metadata Storage

**Approach:** Use Polars native metadata API (write_parquet `metadata` parameter)

**Metadata keys:**
- `pdstools:source_file` - Full path to the original source file
- `pdstools:sample_percentage` - Percentage of interactions sampled (as string)
- `pdstools:sample_percentage_method` - Either "exact" or "approximated"

**Writing:**
```python
metadata = {
    'pdstools:source_file': '/path/to/original.parquet',
    'pdstools:sample_percentage': '10.5',
    'pdstools:sample_percentage_method': 'exact'
}
df.write_parquet(output_path, metadata=metadata)
```

**Reading:**
```python
metadata = pl.read_parquet_metadata('decision_analyzer_sample_100k.parquet')
source = metadata.get('pdstools:source_file')
sample_pct = metadata.get('pdstools:sample_percentage')
method = metadata.get('pdstools:sample_percentage_method')
```

### 3. Sample Percentage Calculation Strategy

**Priority order:**

1. **Fraction-based sampling** (e.g., `--sample 10%`):
   - Percentage = `fraction * 100`
   - Method: "exact"
   - No additional calculation needed

2. **Count-based sampling with existing total** (e.g., `--sample 100000`):
   - The `sample_and_save` function already counts unique interactions to check if sampling is needed
   - Percentage = `(n / total) * 100`
   - Method: "exact"
   - Reuse the existing count, no extra work

3. **File size approximation** (fallback for edge cases):
   - Only when input file is ≥500MB AND we somehow don't have the exact count
   - Percentage ≈ `(output_file_size / input_file_size) * 100`
   - Method: "approximated"
   - Note: This should rarely happen given that count-based sampling already calculates totals

**Performance threshold:**
- 500 MB: chosen as the balance point between accuracy and performance
- Most files under 500MB can be scanned for unique interaction counts quickly

### 4. Metadata Inheritance from Pre-Sampled Files

**Detection:**
1. Check if `source_path` points to an existing parquet file
2. Try to read metadata: `pl.read_parquet_metadata(source_path)`
3. Check for `pdstools:source_file` key

**If source is already sampled:**
- Inherit `pdstools:source_file` value (don't use the intermediate file path)
- Multiply percentages: `existing_pct * new_pct / 100`
- Inherit the most conservative method ("approximated" if either step was approximated)

**Example lineage chain:**
```
Original: exports/data.parquet (1M interactions)
    ↓ 50% sample
decision_analyzer_sample_500k.parquet
    metadata:
      source_file: exports/data.parquet
      sample_percentage: 50
      sample_percentage_method: exact
    ↓ 20% sample
decision_analyzer_sample_100k.parquet
    metadata:
      source_file: exports/data.parquet  (inherited)
      sample_percentage: 10  (50% × 20% = 10%)
      sample_percentage_method: exact
```

**Error handling:**
- If metadata read fails, treat as fresh data
- If metadata keys are missing, treat as fresh data
- Log warnings for unexpected metadata formats

### 5. Function Signature Changes

**Modified `sample_and_save` signature:**
```python
def sample_and_save(
    df: pl.LazyFrame,
    n: int | None = None,
    fraction: float | None = None,
    output_dir: str | None = None,
    source_path: str | None = None,  # NEW: path to original file
) -> tuple[pl.LazyFrame, Path | None]:
```

**New utility function:**
```python
def format_count_for_filename(count: int) -> str:
    """Format an interaction count for use in filenames.

    Uses human-readable abbreviations with 2 significant figures.

    Examples:
        87 → "87"
        1500 → "1.5k"
        87432 → "87k"
        1234567 → "1.2M"
    """
```

## Implementation Flow

### Updated `sample_and_save` Flow

1. **Determine source information:**
   - If `source_path` provided, check for existing metadata
   - If source has metadata, extract original source and percentage
   - If not, use `source_path` as the original source

2. **Perform sampling:**
   - Call existing `sample_interactions()` logic
   - Collect the sampled DataFrame

3. **Calculate sample percentage:**
   - If `fraction` provided: percentage = `fraction * 100` (exact)
   - If `n` provided: percentage = `(n / total) * 100` where `total` is from the existing count check (exact)
   - If source had metadata: multiply existing percentage by new percentage
   - Store calculation method ("exact" or "approximated")

4. **Determine output filename:**
   - Count unique interactions in sampled data: `sampled_count`
   - Format count: `formatted = format_count_for_filename(sampled_count)`
   - Build filename: `decision_analyzer_sample_{formatted}.parquet`

5. **Build metadata:**
   ```python
   metadata = {
       'pdstools:source_file': original_source,
       'pdstools:sample_percentage': f'{final_percentage:.2f}',
       'pdstools:sample_percentage_method': method
   }
   ```

6. **Write with metadata:**
   ```python
   sampled_df.write_parquet(out_path, metadata=metadata)
   ```

7. **Return LazyFrame and path:**
   - Scan the written file and return with path

### CLI Integration Points

The CLI invocation in `Home.py` (lines 105-110) will need to pass the source path:

```python
raw_data, sample_path = sample_and_save(
    raw_data,
    n=sample_kwargs.get("n"),
    fraction=sample_kwargs.get("fraction"),
    output_dir=get_temp_dir(),
    source_path=configured_path or uploaded_file_path,  # NEW
)
```

## Testing Strategy

1. **Unit tests for `format_count_for_filename`:**
   - Test all size ranges (units, k, M, B)
   - Test boundary conditions (999, 1000, 999999, 1000000, etc.)
   - Test 2 significant figure rounding

2. **Integration tests for `sample_and_save`:**
   - Test filename generation with various counts
   - Test metadata writing and reading
   - Test lineage tracking with chained sampling
   - Test both fraction and count-based sampling
   - Test with and without source_path parameter

3. **Edge case tests:**
   - Very small samples (<10 interactions)
   - Very large samples (billions)
   - Invalid source paths
   - Corrupted metadata
   - Sampling a sample multiple times

## Backward Compatibility

- Existing code calling `sample_and_save` without `source_path` will continue to work
- Old sample files without metadata will be treated as fresh data if used as input
- The function signature is backward compatible (new parameter is optional)

## Future Enhancements

1. Add a utility function to read and display metadata from sampled files
2. Consider adding timestamp metadata
3. Consider adding sampling method (hash-based deterministic) to metadata
4. Add CLI flag to display metadata: `pdstools da --show-metadata <file>`

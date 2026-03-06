# Enhanced Progress Feedback for Large Dataset Loading

## Problem Statement

When loading very large datasets (e.g., 4.5 GB zipped, hive-partitioned parquet files from NAB), the Decision Analyzer UI displays "Loaded data from configured path" but then shows no updates for an extended period while the Streamlit activity icon rotates. The UI eventually updates to show "Sampling" with a rotating icon, but users have no indication of how long these operations will take.

This creates a poor user experience where users are uncertain whether:
- The application is still working
- How long they should wait
- Whether something has gone wrong

## Goals

1. Provide time estimates for long-running data loading operations
2. Focus on the two main bottlenecks:
   - Extracting large zip archives
   - Data sampling operations
3. Use simple, maintainable solution with minimal code changes
4. Keep estimates conservative to avoid user frustration

## Non-Goals

- Real-time progress bars with exact percentages
- Progress tracking for all operations (only the slowest ones)
- Multi-threaded or complex progress monitoring
- CLI progress feedback (Streamlit UI only for now)

## Proposed Solution

### Approach: Enhanced Streamlit Spinners with File Size-Based Time Estimates

Enhance existing `st.spinner()` calls to display estimated time ranges based on file size. Use calibrated extraction speeds to provide reasonable time estimates without complex progress tracking.

**Example messages:**
- "Extracting archive... (estimated: 2 to 4 minutes)"
- "Sampling 50,000 interactions... (estimated: 30 seconds)"

### Why This Approach?

**Advantages:**
- Minimal code changes - works within existing Streamlit patterns
- Simple to implement and maintain
- No complex threading or progress monitoring
- Good enough - users just need rough guidance
- Falls back gracefully if estimation fails

**Alternatives Considered:**

1. **tqdm Integration** - Real progress tracking but adds dependency complexity and requires custom extraction logic
2. **Hybrid with Threading** - Most accurate but too complex and prone to bugs with Streamlit's execution model
3. **Progress Bars** - More visual but requires actual progress tracking, not just estimates

## Architecture

### New Components

**`python/pdstools/utils/progress_utils.py`** - New utility module containing:
- `estimate_extraction_time(file_size_bytes: int) -> tuple[float, float]`
- `estimate_sampling_time(row_count: int, sample_size: int) -> tuple[float, float]`
- `format_time_estimate(min_sec: float, max_sec: float) -> str`

### Modified Components

**`python/pdstools/app/decision_analyzer/da_streamlit_utils.py`**
- Update `handle_data_path()` around line 438 to add time estimates for zip extraction

**`python/pdstools/app/decision_analyzer/Home.py`**
- Enhance sampling spinner message around line 117 to include time estimates

### New Dependency

- `humanize` - For user-friendly time formatting ("2 minutes" vs "120 seconds")

## Detailed Design

### Time Estimation Logic

```python
def estimate_extraction_time(file_size_bytes: int) -> tuple[float, float]:
    """
    Estimate extraction time for a zip file based on size.

    Uses calibrated extraction speeds to provide min/max range.
    Conservative estimates to avoid under-promising.

    Returns:
        tuple of (min_seconds, max_seconds) for a range estimate
    """
    # Calibrated extraction speeds (MB/s)
    FAST_SPEED_MBS = 100  # SSD with good CPU
    SLOW_SPEED_MBS = 30   # HDD or compressed data

    size_mb = file_size_bytes / (1024 * 1024)
    min_time = size_mb / FAST_SPEED_MBS
    max_time = size_mb / SLOW_SPEED_MBS

    return (min_time, max_time)

def format_time_estimate(min_sec: float, max_sec: float) -> str:
    """Format time range as user-friendly string using humanize."""
    import humanize

    if max_sec < 10:
        return "a few seconds"
    elif max_sec < 60:
        return f"{int(max_sec)} seconds"
    else:
        # Show range for longer operations
        min_str = humanize.naturaldelta(min_sec)
        max_str = humanize.naturaldelta(max_sec)
        if min_str == max_str:
            return min_str
        return f"{min_str} to {max_str}"
```

### Integration Points

**Zip Extraction** (`handle_data_path()`):
```python
if p.is_file() and p.suffix.lower() == ".zip":
    try:
        file_size = p.stat().st_size
        min_time, max_time = estimate_extraction_time(file_size)
        time_msg = format_time_estimate(min_time, max_time)
        spinner_msg = f"Extracting archive... (estimated: {time_msg})"
    except Exception:
        spinner_msg = "Extracting archive..."

    with st.spinner(spinner_msg):
        # existing extraction code
```

**Data Sampling** (`Home.py`):
```python
# Estimate based on row count if available
try:
    row_count = get_approximate_row_count(raw_data)
    sample_time = estimate_sampling_time(row_count, sample_kwargs)
    time_msg = format_time_estimate(*sample_time)
    sampling_msg = f"Sampling {label}... (estimated: {time_msg})"
except Exception:
    # Fall back to existing message
    sampling_msg = f"Sampling {label}..."
```

## Error Handling & Edge Cases

### File Size Unavailable
- If `p.stat().st_size` fails or returns 0, fall back to simple spinner without estimate
- Log issue for debugging but don't fail operation

### Estimation Accuracy
- Use time ranges ("2 to 4 minutes") to account for variability
- Keep estimates conservative (slower rather than faster)
- If operation completes faster, users see pleasant surprise
- If slower, range helps manage expectations

### Very Large Files (>10 GB)
- Cap upper time estimate at reasonable values (e.g., "10 to 15 minutes")
- Consider adding note: "Large file detected - this may take several minutes"

### Network/Slow Storage
- Estimates assume local storage
- Actual time may exceed estimate on network mounts or slow drives
- Acceptable - estimates are guidance, not guarantees

### Sampling Time Estimation
- Estimate based on approximate row count from lazy frame metadata if available
- Fall back to file size if row count unavailable
- Fall back to generic message if neither available

## Testing Strategy

### Unit Tests (`test_progress_utils.py`)

Test the calculation logic only (no real files):

```python
def test_estimate_extraction_time_small_file():
    # 10 MB file (just a number, not real file)
    min_sec, max_sec = estimate_extraction_time(10 * 1024 * 1024)
    assert min_sec < 1
    assert max_sec < 5

def test_estimate_extraction_time_large_file():
    # 4.5 GB file (just a number)
    min_sec, max_sec = estimate_extraction_time(4.5 * 1024 * 1024 * 1024)
    assert 45 < min_sec < 60  # ~45s at 100 MB/s
    assert 150 < max_sec < 180  # ~150s at 30 MB/s

def test_format_time_estimate_seconds():
    result = format_time_estimate(5, 8)
    assert "second" in result.lower()

def test_format_time_estimate_minutes():
    result = format_time_estimate(120, 180)
    assert "minute" in result.lower()

def test_format_time_estimate_range():
    result = format_time_estimate(60, 240)
    assert "to" in result
```

### Integration Testing

Manual testing with real datasets:
- Test with 4.5 GB NAB dataset to verify estimates are reasonable
- Test with various file sizes (small, medium, large)
- Verify spinner messages appear correctly
- Check that fallback works when estimation fails

### Backward Compatibility

- All existing code paths remain unchanged
- If time estimation fails, falls back to existing simple spinners
- No API changes, purely UI enhancement
- No breaking changes

## Implementation Rollout

### Phase 1 - Core Functionality
1. Create `progress_utils.py` with estimation functions
2. Add `humanize` dependency to `pyproject.toml`
3. Update `handle_data_path()` for zip extraction time estimates
4. Write unit tests for `progress_utils.py`

### Phase 2 - Sampling Enhancement
5. Add sampling time estimation helper
6. Update `Home.py` to show sampling time estimates
7. Test with various dataset sizes

### Phase 3 - Calibration
8. Test with real large files (NAB dataset)
9. Adjust speed constants if estimates are significantly off
10. Document expected accuracy in code comments

## Success Metrics

### Qualitative
- Users report better experience with large file loading
- Reduced uncertainty about whether application is working
- Fewer support questions about "is it still loading?"

### Quantitative
- Time estimates are within 50% accuracy for typical cases
  - 4.5 GB file estimated at "2-4 minutes" should complete in 1.5-6 minutes
- No regression in loading time (overhead negligible)
- No new errors or crashes from estimation code

## Future Enhancements

Potential improvements for future iterations (not in scope now):

1. **Real Progress Tracking** - Use tqdm for actual progress bars
2. **Adaptive Calibration** - Learn typical speeds from past loads
3. **CLI Progress** - Extend progress feedback to CLI usage
4. **Progress for All Operations** - Add estimates for other slow operations (directory scanning, initial processing)
5. **Cancellation** - Allow users to cancel long-running operations

## Dependencies

- Add `humanize` library (lightweight, well-maintained)
- No other new dependencies required

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Time estimates significantly inaccurate | Use conservative ranges; calibrate with real data |
| Code adds noticeable overhead | Keep estimation logic simple; benchmark if needed |
| `humanize` dependency issues | Library is stable and widely used; has fallback logic |
| Works poorly on slow storage | Set expectations via conservative estimates and ranges |
| Future Streamlit changes break spinners | Minimal code changes reduce maintenance burden |

## Open Questions

None - design is complete and approved.

## References

- Issue: User feedback about unclear progress for large NAB dataset (4.5 GB)
- Affected files: `DecisionAnalyzer.py`, `da_streamlit_utils.py`, `Home.py`, `data_read_utils.py`
- Related: Previous work on data sampling and caching features

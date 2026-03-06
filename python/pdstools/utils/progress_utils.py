"""Utilities for progress feedback and time estimation.

This module provides functions to estimate operation times and format them
in user-friendly ways. Used primarily by the Decision Analysis Tool Streamlit
app to show progress feedback for long-running operations like:

- Extracting large zip archives
- Sampling large datasets

The estimates are based on calibrated speeds and provide ranges to account
for system variability.
"""


def estimate_extraction_time(file_size_bytes: int) -> tuple[float, float]:
    """Estimate extraction time for a zip file based on size.

    Uses calibrated extraction speeds to provide min/max range.
    Conservative estimates to avoid under-promising.

    Parameters
    ----------
    file_size_bytes : int
        Size of the file in bytes

    Returns
    -------
    tuple[float, float]
        (min_seconds, max_seconds) for a range estimate

    Examples
    --------
    >>> min_time, max_time = estimate_extraction_time(1024 * 1024 * 1024)  # 1 GB
    >>> min_time < max_time
    True
    """
    # Calibrated extraction speeds (MB/s)
    FAST_SPEED_MBS = 100  # SSD with good CPU
    SLOW_SPEED_MBS = 30  # HDD or compressed data

    size_mb = file_size_bytes / (1024 * 1024)
    min_time = size_mb / FAST_SPEED_MBS
    max_time = size_mb / SLOW_SPEED_MBS

    return (min_time, max_time)


def format_time_estimate(min_sec: float, max_sec: float) -> str:
    """Format time range as user-friendly string.

    Uses humanize library to create natural language time descriptions.
    Shows ranges for operations over 10 seconds, simple descriptions for
    shorter operations.

    Parameters
    ----------
    min_sec : float
        Minimum estimated time in seconds
    max_sec : float
        Maximum estimated time in seconds

    Returns
    -------
    str
        User-friendly time description

    Examples
    --------
    >>> format_time_estimate(2, 5)
    'a few seconds'

    >>> format_time_estimate(120, 180)
    '2 minutes to 3 minutes'
    """
    try:
        import humanize
    except ImportError:
        # Fallback if humanize not available
        if max_sec < 60:
            return f"{int(max_sec)} seconds"
        else:
            return f"{int(max_sec / 60)} minutes"

    if max_sec < 10:
        return "a few seconds"
    elif max_sec < 60:
        return f"{int(max_sec)} seconds"
    else:
        min_str = humanize.naturaldelta(min_sec)
        max_str = humanize.naturaldelta(max_sec)
        if min_str == max_str:
            return min_str
        return f"{min_str} to {max_str}"


def estimate_sampling_time(total_rows: int, sample_size: int) -> tuple[float, float]:
    """Estimate time for sampling operations based on dataset size.

    Parameters
    ----------
    total_rows : int
        Total number of rows in the dataset
    sample_size : int
        Target sample size

    Returns
    -------
    tuple[float, float]
        (min_seconds, max_seconds) for a range estimate

    Examples
    --------
    >>> min_time, max_time = estimate_sampling_time(1_000_000, 50_000)
    >>> min_time < max_time
    True
    """
    # Calibrated sampling speeds (rows/second)
    # Based on Polars hash-based sampling performance
    FAST_ROWS_PER_SEC = 1_000_000  # Good CPU, in-memory data
    SLOW_ROWS_PER_SEC = 100_000  # Slower system or disk-based

    # If no sampling needed (data smaller than sample), very fast
    if total_rows <= sample_size:
        return (0.1, 0.5)

    # Estimate based on total rows (not sample size) since we scan all data
    min_time = total_rows / FAST_ROWS_PER_SEC
    max_time = total_rows / SLOW_ROWS_PER_SEC

    return (min_time, max_time)

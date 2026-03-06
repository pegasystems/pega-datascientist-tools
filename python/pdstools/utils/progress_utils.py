"""Utilities for progress feedback and time estimation."""


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

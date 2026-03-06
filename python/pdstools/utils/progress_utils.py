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

from pdstools.utils.progress_utils import estimate_extraction_time


def test_estimate_extraction_time_small_file():
    """Test estimation for 10 MB file."""
    min_sec, max_sec = estimate_extraction_time(10 * 1024 * 1024)
    assert min_sec < 1  # Should be very fast
    assert max_sec < 5
    assert min_sec < max_sec


def test_estimate_extraction_time_large_file():
    """Test estimation for 4.5 GB file."""
    min_sec, max_sec = estimate_extraction_time(4.5 * 1024 * 1024 * 1024)
    assert 45 < min_sec < 60  # ~45 seconds at 100 MB/s
    assert 150 < max_sec < 180  # ~150 seconds at 30 MB/s
    assert min_sec < max_sec


def test_estimate_extraction_time_medium_file():
    """Test estimation for 500 MB file."""
    min_sec, max_sec = estimate_extraction_time(500 * 1024 * 1024)
    assert min_sec > 0
    assert max_sec > min_sec

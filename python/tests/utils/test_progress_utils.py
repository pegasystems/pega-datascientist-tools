from pdstools.utils.progress_utils import (
    estimate_extraction_time,
    estimate_sampling_time,
    format_time_estimate,
)


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


def test_format_time_estimate_very_short():
    """Test formatting for operations under 10 seconds."""
    result = format_time_estimate(2, 5)
    assert result == "a few seconds"


def test_format_time_estimate_seconds():
    """Test formatting for operations under a minute."""
    result = format_time_estimate(15, 45)
    assert "second" in result.lower()
    assert "45" in result


def test_format_time_estimate_minutes():
    """Test formatting for operations over a minute."""
    result = format_time_estimate(120, 180)
    assert "minute" in result.lower()


def test_format_time_estimate_range():
    """Test that longer operations show a range."""
    result = format_time_estimate(60, 240)
    assert "to" in result  # Should show "X to Y"


def test_format_time_estimate_same_range():
    """Test that similar times don't show redundant range."""
    result = format_time_estimate(120, 130)
    result_count = result.count("minute")
    assert result_count <= 2  # At most "X minutes to Y minutes"


def test_estimate_sampling_time_small():
    """Test estimation for small dataset."""
    # 10k rows, sampling 5k
    min_sec, max_sec = estimate_sampling_time(10_000, 5_000)
    assert min_sec < max_sec
    assert max_sec < 10  # Should be fast


def test_estimate_sampling_time_large():
    """Test estimation for large dataset."""
    # 10M rows, sampling 50k
    min_sec, max_sec = estimate_sampling_time(10_000_000, 50_000)
    assert min_sec < max_sec
    assert min_sec > 0


def test_estimate_sampling_time_no_sampling():
    """Test when no sampling needed (data smaller than sample size)."""
    min_sec, max_sec = estimate_sampling_time(1_000, 50_000)
    assert min_sec < max_sec
    assert max_sec < 1  # Very fast


def test_format_time_estimate_without_humanize(monkeypatch):
    """Test fallback when humanize not available."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "humanize":
            raise ImportError("humanize not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Should still work with fallback
    result = format_time_estimate(30, 45)
    assert "45" in result or "second" in result.lower()

    result = format_time_estimate(120, 180)
    assert "minute" in result.lower()


def test_estimate_extraction_time_zero_size():
    """Test with zero-byte file."""
    min_sec, max_sec = estimate_extraction_time(0)
    assert min_sec == 0
    assert max_sec == 0


def test_estimate_sampling_time_zero_rows():
    """Test with empty dataset."""
    min_sec, max_sec = estimate_sampling_time(0, 1000)
    assert min_sec >= 0
    assert max_sec >= 0

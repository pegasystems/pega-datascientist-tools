from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone

import pytest
from pdstools.sample_data._calendar import SampleCalendar, sample_calendar


def test_default_canonical_window():
    window = sample_calendar(end="canonical")
    assert (window.days, window.start, window.end, window.offset) == (
        182,
        date(2000, 1, 3),
        date(2000, 7, 2),
        timedelta(0),
    )


@pytest.mark.parametrize(
    ("days", "start"),
    [(56, date(2000, 5, 8)), (60, date(2000, 5, 4)), (70, date(2000, 4, 24))],
)
def test_exact_lengths_include_partial_first_week(days, start):
    window = sample_calendar(days, end="canonical")
    assert window.start == start
    assert (window.end - window.start).days + 1 == days


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 11, 5), date(2026, 11, 1)),
        (date(2026, 11, 8), date(2026, 11, 1)),
        (date(2026, 11, 9), date(2026, 11, 8)),
    ],
)
def test_default_uses_completed_sunday(today, expected):
    window = sample_calendar(today=today)
    assert window.end == expected


@pytest.mark.parametrize("end", [date(2026, 11, 5), date(2026, 11, 1)])
def test_explicit_end_normalizes_inclusively(end):
    window = sample_calendar(60, end)
    assert window.end == date(2026, 11, 1)
    assert window.start == date(2026, 9, 3)
    assert window.offset.days % 7 == 0


def test_date_shift_preserves_every_weekday():
    canonical = sample_calendar(60, "canonical")
    shifted = sample_calendar(60, date(2026, 11, 5))
    for day in range(60):
        before = canonical.start + timedelta(days=day)
        after = shifted.start + timedelta(days=day)
        assert after - before == shifted.offset
        assert after.weekday() == before.weekday()


@pytest.mark.parametrize("days", [0, -1, 5, 30, 55])
def test_reject_short_windows(days):
    with pytest.raises(ValueError, match="at least 56"):
        sample_calendar(days, "canonical")


@pytest.mark.parametrize("days", [True, 56.0, "56", None])
def test_reject_noninteger_windows(days):
    with pytest.raises(TypeError, match="integer"):
        sample_calendar(days, "canonical")


@pytest.mark.parametrize("end", ["2026-11-05", 1, datetime(2026, 11, 5, tzinfo=timezone.utc)])
def test_reject_invalid_end_types(end):
    with pytest.raises(TypeError, match="end must be"):
        sample_calendar(end=end)


def test_reject_non_normalized_calendar():
    with pytest.raises(ValueError, match="Sunday"):
        SampleCalendar(182, date(2026, 11, 5))
    with pytest.raises(TypeError, match="without a time"):
        SampleCalendar(182, datetime(2026, 11, 1, tzinfo=timezone.utc))


def test_reject_unrepresentable_window():
    with pytest.raises(ValueError, match="supported date range"):
        sample_calendar(10**12, "canonical")
    with pytest.raises(ValueError, match="Sunday-aligned"):
        sample_calendar(end=date.min)


def test_calendar_is_immutable():
    window = sample_calendar(end="canonical")
    with pytest.raises(FrozenInstanceError):
        window.days = 60

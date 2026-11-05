"""Canonical dates shared by all synthetic sample views."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

CANONICAL_END = date(2000, 7, 2)
DEFAULT_DAYS = 182
MINIMUM_DAYS = 56

EndDate = date | Literal["canonical"] | None


@dataclass(frozen=True)
class SampleCalendar:
    """Describe an exact window and its weekday-preserving date shift.

    Parameters
    ----------
    days : int
        Number of dates in the window, at least 56.
    end : date
        Normalized Sunday ending the requested window.
    """

    days: int
    end: date

    def __post_init__(self) -> None:
        if isinstance(self.days, bool) or not isinstance(self.days, int):
            raise TypeError("days must be an integer.")
        if self.days < MINIMUM_DAYS:
            raise ValueError(f"days must be at least {MINIMUM_DAYS}; shorter sample windows are not supported.")
        if not isinstance(self.end, date) or isinstance(self.end, datetime):
            raise TypeError("end must be a date without a time component.")
        if self.end.weekday() != 6:
            raise ValueError("The normalized sample end must be a Sunday.")
        try:
            _ = self.canonical_start, self.start
        except OverflowError as exc:
            raise ValueError("days exceeds the supported date range for this sample window.") from exc

    @property
    def canonical_start(self) -> date:
        """First canonical date, retaining the exact requested length."""
        return CANONICAL_END - timedelta(days=self.days - 1)

    @property
    def start(self) -> date:
        """First shifted date, including a partial first week when needed."""
        return self.end - timedelta(days=self.days - 1)

    @property
    def offset(self) -> timedelta:
        """Whole-week offset from the canonical calendar."""
        return self.end - CANONICAL_END


def sample_calendar(
    days: int = DEFAULT_DAYS,
    end: EndDate = None,
    *,
    today: date | None = None,
) -> SampleCalendar:
    """Resolve the common sample calendar without changing its length.

    Parameters
    ----------
    days : int, default 182
        Requested number of dates; the minimum is 56.
    end : date, "canonical" or None, default None
        Explicit dates resolve to the last Sunday on or before that date.
        ``None`` uses the most recent completed Sunday in UTC, excluding today.
        ``"canonical"`` uses Sunday 2000-07-02.
    today : date or None, default None
        Reference date for resolving the default end, used for deterministic
        internal checks. Ignored when ``end`` is supplied.

    Returns
    -------
    SampleCalendar
        Exact-length window with a whole-week shift.
    """
    if end == "canonical":
        return SampleCalendar(days, CANONICAL_END)
    try:
        if end is None:
            anchor = (today if today is not None else datetime.now(timezone.utc).date()) - timedelta(days=1)
        elif isinstance(end, date) and not isinstance(end, datetime):
            anchor = end
        else:
            raise TypeError('end must be a date, "canonical", or None.')
        sunday = anchor - timedelta(days=(anchor.weekday() + 1) % 7)
    except OverflowError as exc:
        raise ValueError("end is outside the supported range for a Sunday-aligned sample window.") from exc
    return SampleCalendar(days, sunday)

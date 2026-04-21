"""Date and time helpers for Pega-formatted timestamps."""

import datetime

import polars as pl
from polars._typing import PolarsTemporalType


def parse_pega_date_time_formats(
    timestamp_col="SnapshotTime",
    timestamp_fmt: str | None = None,
    timestamp_dtype: PolarsTemporalType = pl.Datetime,
) -> pl.Expr:
    """Parses Pega DateTime formats.

    Supports commonly used formats:

    - "%Y-%m-%d %H:%M:%S"
    - "%Y%m%dT%H%M%S.%f %Z"
    - "%d-%b-%y"
    - "%d%b%Y:%H:%M:%S"
    - "%Y%m%d"

    Removes timezones, and rounds to seconds, with a 'ns' time unit.

    In the implementation, the last expression uses timestamp_fmt or %Y.
    This is a bit of a hack, because if we pass None, it tries to infer automatically.
    Inferring raises when it can't find an appropriate format, so that's not good.

    Parameters
    ----------
    timestampCol: str, default = 'SnapshotTime'
        The column to parse
    timestamp_fmt: str, default = None
        An optional format to use rather than the default formats
    timestamp_dtype: PolarsTemporalType, default = pl.Datetime
        The data type to convert into. Can be either Date, Datetime, or Time.

    """
    result = pl.coalesce(
        pl.col(timestamp_col).str.strptime(
            timestamp_dtype,
            "%Y-%m-%d %H:%M:%S",
            strict=False,
            ambiguous="null",
        ),
        pl.col(timestamp_col).str.strptime(
            timestamp_dtype,
            "%Y%m%dT%H%M%S.%3f %Z",
            strict=False,
            ambiguous="null",
        ),
        pl.col(timestamp_col).str.strptime(
            timestamp_dtype,
            "%d%b%Y:%H:%M:%S",
            strict=False,
            ambiguous="null",
        ),
        pl.col(timestamp_col).str.slice(0, 8).str.strptime(timestamp_dtype, "%Y%m%d", strict=False, ambiguous="null"),
        pl.col(timestamp_col).str.strptime(
            timestamp_dtype,
            "%d-%b-%y",
            strict=False,
            ambiguous="null",
        ),
        pl.col(timestamp_col).str.strptime(
            timestamp_dtype,
            timestamp_fmt or "%Y",
            strict=False,
            ambiguous="null",
        ),
    )

    if timestamp_dtype != pl.Date:
        result = result.dt.replace_time_zone(None).dt.cast_time_unit("ns")

    return result


def from_prpc_date_time(
    x: str,
    return_string: bool = False,
    use_timezones: bool = True,
) -> datetime.datetime | str:
    """Convert from a Pega date-time string.

    Parameters
    ----------
    x: str
        String of Pega date-time
    return_string: bool, default=False
        If True it will return the date in string format. If
        False it will return in datetime type

    Returns
    -------
    datetime.datetime | str
        The converted date in datetime format or string.

    Examples
    --------
        >>> fromPRPCDateTime("20180316T134127.847 GMT")
        >>> fromPRPCDateTime("20180316T134127.847 GMT", True)
        >>> fromPRPCDateTime("20180316T184127.846")
        >>> fromPRPCDateTime("20180316T184127.846", True)

    """
    from zoneinfo import ZoneInfo

    timezonesplits = x.split(" ")

    if len(timezonesplits) > 1:
        x = timezonesplits[0]

    if "." in x:
        date_no_frac, frac_sec = x.split(".")
        if len(frac_sec) > 3:
            frac_sec = frac_sec[:3]
        elif len(frac_sec) < 3:
            frac_sec = f"{int(frac_sec):<03d}"
    else:
        date_no_frac = x

    dt = datetime.datetime.strptime(date_no_frac, "%Y%m%dT%H%M%S")

    if use_timezones and len(timezonesplits) > 1:
        dt = dt.replace(tzinfo=ZoneInfo(timezonesplits[1]))

    if "." in x:
        dt = dt.replace(microsecond=int(frac_sec))

    if return_string:
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return dt


def to_prpc_date_time(dt: datetime.datetime) -> str:
    """Convert to a Pega date-time string

    Parameters
    ----------
    x: datetime.datetime
        A datetime object

    Returns
    -------
    str
        A string representation in the format used by Pega

    Examples
    --------
        >>> toPRPCDateTime(datetime.datetime.now())

    """
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.strftime("%Y%m%dT%H%M%S.%f")[:-3] + dt.strftime(" GMT%z")


def _get_start_end_date_args(
    data: pl.Series | pl.LazyFrame | pl.DataFrame,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
    window: int | datetime.timedelta | None = None,
    datetime_field="SnapshotTime",
):
    if isinstance(data, pl.DataFrame):
        data_min_date = data.select(pl.col(datetime_field).min()).item()
        data_max_date = data.select(pl.col(datetime_field).max()).item()
    elif isinstance(data, pl.LazyFrame):
        data_min_date = data.select(pl.col(datetime_field).min()).collect().item()
        data_max_date = data.select(pl.col(datetime_field).max()).collect().item()
    else:  # pl.Series
        data_min_date = data.min()
        data_max_date = data.max()

    if window is not None and not isinstance(window, datetime.timedelta):
        window = datetime.timedelta(days=window)

    if start_date and end_date and window:
        raise ValueError(
            "Only max two of 'start_date', 'end_date' or 'window_days' can be set",
        )
    if not end_date:
        if window is None or start_date is None:
            end_date = data_max_date
        else:
            end_date = start_date + window - datetime.timedelta(days=1)
    if not start_date:
        if window is None or end_date is None:
            start_date = data_min_date
        else:
            start_date = end_date - window + datetime.timedelta(days=1)

    if start_date and end_date and start_date > end_date:
        raise ValueError(
            f"The start date {start_date} should be before the end date {end_date}",
        )

    return start_date, end_date

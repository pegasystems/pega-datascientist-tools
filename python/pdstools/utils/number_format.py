"""Number formatting utilities inspired by great_tables fmt_number.

This module provides a `NumberFormat` class that defines how numbers should be
formatted, with a subset of parameters from great_tables' `fmt_number`. Formats
can be translated to different output targets (great_tables, pandas Styler,
Polars expressions).

See: https://posit-dev.github.io/great-tables/reference/vals.fmt_number.html
"""

from dataclasses import dataclass
from math import isnan
from typing import TYPE_CHECKING, Callable, Optional, Union

import polars as pl

if TYPE_CHECKING:
    from great_tables import GT

__all__ = ["NumberFormat"]

# Type alias for format value input
NumericValue = Union[int, float, None]


@dataclass(frozen=True)
class NumberFormat:
    """Specification for formatting numeric values.

    Inspired by great_tables' `fmt_number`, this immutable class defines
    formatting options that translate to multiple output targets.

    Parameters
    ----------
    decimals : int, default 0
        Number of decimal places.
    scale_by : float, default 1.0
        Multiplier applied before formatting (use 100 for percentages).
    compact : bool, default False
        Use compact notation (K, M, B, T).
    locale : str, default "en_US"
        Locale for separators. Supports "en_US" and "de_DE".
    suffix : str, default ""
        String appended after the number (e.g., "%").

    Examples
    --------
    >>> NumberFormat(decimals=2).format_value(1234.567)
    '1,234.57'

    >>> NumberFormat(decimals=1, scale_by=100, suffix="%").format_value(0.1234)
    '12.3%'

    >>> NumberFormat(compact=True).format_value(1234567)
    '1M'
    """

    decimals: int = 0
    scale_by: float = 1.0
    compact: bool = False
    locale: str = "en_US"
    suffix: str = ""

    def format_value(self, value: NumericValue) -> str:
        """Format a single numeric value.

        Parameters
        ----------
        value
            The value to format. Returns "" for None or NaN.

        Returns
        -------
        str
            Formatted string representation.
        """
        if value is None or (isinstance(value, float) and isnan(value)):
            return ""

        try:
            num = float(value) * self.scale_by
        except (ValueError, TypeError):
            return str(value)

        formatted = (
            self._format_compact(num) if self.compact else self._format_standard(num)
        )
        return formatted + self.suffix

    def _format_standard(self, num: float) -> str:
        """Format with locale-aware thousand/decimal separators."""
        formatted = f"{num:,.{self.decimals}f}"
        if self.locale == "de_DE":
            # German: swap comma/dot
            formatted = (
                formatted.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
            )
        return formatted

    def _format_compact(self, num: float) -> str:
        """Format in compact notation (K/M/B/T)."""
        if num == 0:
            return "0"

        abs_num = abs(num)
        for threshold, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
            if abs_num >= threshold:
                return f"{num / threshold:,.{self.decimals}f}{suffix}"
        return f"{num:,.{self.decimals}f}"

    def to_pandas_format(self) -> Union[str, Callable[[NumericValue], str]]:
        """Convert to pandas Styler format specification.

        Returns
        -------
        str or callable
            Format string or callable for `pandas.Styler.format()`.
        """
        # Use callable for complex formatting (compact, non-percentage suffix)
        if self.compact or (
            self.suffix and not (self.scale_by == 100 and self.suffix == "%")
        ):
            return self.format_value

        # Native pandas percentage formatting
        if self.scale_by == 100 and self.suffix == "%":
            return f"{{:.{self.decimals}%}}"

        return f"{{:,.{self.decimals}f}}"

    def apply_to_gt(self, gt: "GT", columns: list[str]) -> "GT":
        """Apply formatting to a great_tables GT object.

        Parameters
        ----------
        gt
            The GT object to format.
        columns
            Column names to apply formatting to.

        Returns
        -------
        GT
            The GT object with formatting applied.
        """
        if self.scale_by == 100 and self.suffix == "%":
            return gt.fmt_percent(decimals=self.decimals, columns=columns)
        if self.compact:
            return gt.fmt_number(decimals=self.decimals, compact=True, columns=columns)
        if self.scale_by != 1.0:
            return gt.fmt_number(
                decimals=self.decimals, scale_by=self.scale_by, columns=columns
            )
        return gt.fmt_number(decimals=self.decimals, columns=columns)

    def to_polars_expr(self, column: str) -> pl.Expr:
        """Create a Polars expression that formats a column as strings.

        Parameters
        ----------
        column
            Column name to format.

        Returns
        -------
        pl.Expr
            Expression producing formatted string values.
        """
        return pl.col(column).map_elements(self.format_value, return_dtype=pl.Utf8)

    def format_polars_column(
        self, df: pl.DataFrame, column: str, output_column: Optional[str] = None
    ) -> pl.DataFrame:
        """Add a formatted string column to a DataFrame.

        Parameters
        ----------
        df
            Source DataFrame.
        column
            Column to format.
        output_column
            Name for the formatted column. Defaults to "{column}_formatted".

        Returns
        -------
        pl.DataFrame
            DataFrame with the formatted column added.
        """
        output_column = output_column or f"{column}_formatted"
        return df.with_columns(self.to_polars_expr(column).alias(output_column))

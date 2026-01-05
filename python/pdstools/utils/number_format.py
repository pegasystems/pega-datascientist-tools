"""Number formatting utilities inspired by great_tables fmt_number.

This module provides a NumberFormat class that defines how numbers should be
formatted, with a subset of parameters from great_tables' fmt_number. The
formats can be translated to different output targets (great_tables, pandas
Styler, Polars expressions, etc.).

See: https://posit-dev.github.io/great-tables/reference/vals.fmt_number.html
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

import polars as pl

if TYPE_CHECKING:
    from great_tables import GT


@dataclass(frozen=True)
class NumberFormat:
    """A specification for how to format numeric values.

    Inspired by great_tables' fmt_number, this class defines formatting options
    that can be translated to different output targets (great_tables, pandas
    Styler, Python f-strings, etc.).

    Parameters
    ----------
    decimals : int, default 0
        Number of decimal places to show.
    scale_by : float, default 1.0
        Value to multiply the number by before formatting.
        Use 100 for percentages (combined with suffix="%").
    compact : bool, default False
        If True, use compact notation (e.g., 1K, 1M, 1B, 1T).
    locale : str, default "en_US"
        Locale for number formatting (affects separators).
        Currently supports "en_US" (comma for thousands, dot for decimal)
        and "de_DE" (dot for thousands, comma for decimal).
    suffix : str, default ""
        String to append after the formatted number (e.g., "%").

    Examples
    --------
    >>> # Simple decimal formatting
    >>> fmt = NumberFormat(decimals=2)
    >>> fmt.format_value(1234.567)
    '1,234.57'

    >>> # Percentage formatting (scale by 100, add % suffix)
    >>> pct_fmt = NumberFormat(decimals=1, scale_by=100, suffix="%")
    >>> pct_fmt.format_value(0.1234)
    '12.3%'

    >>> # Compact notation
    >>> compact_fmt = NumberFormat(compact=True)
    >>> compact_fmt.format_value(1234567)
    '1M'
    """

    decimals: int = 0
    scale_by: float = 1.0
    compact: bool = False
    locale: str = "en_US"
    suffix: str = ""

    def format_value(self, value: Union[int, float, None]) -> str:
        """Format a single value according to this specification.

        Parameters
        ----------
        value : int, float, or None
            The value to format.

        Returns
        -------
        str
            The formatted string representation.
        """
        if value is None or (isinstance(value, float) and value != value):  # NaN check
            return ""

        try:
            num = float(value) * self.scale_by
        except (ValueError, TypeError):
            return str(value)

        if self.compact:
            return self._format_compact(num) + self.suffix

        return self._format_standard(num) + self.suffix

    def _format_standard(self, num: float) -> str:
        """Format number with locale-aware separators."""
        if self.locale == "de_DE":
            # German: dot for thousands, comma for decimal
            formatted = f"{num:,.{self.decimals}f}"
            # Swap separators: comma -> temp, dot -> comma, temp -> dot
            formatted = formatted.replace(",", "_TEMP_")
            formatted = formatted.replace(".", ",")
            formatted = formatted.replace("_TEMP_", ".")
            return formatted
        else:
            # Default (en_US): comma for thousands, dot for decimal
            return f"{num:,.{self.decimals}f}"

    def _format_compact(self, num: float) -> str:
        """Format number in compact notation (K, M, B, T)."""
        if num == 0:
            return "0"

        abs_num = abs(num)
        if abs_num >= 1_000_000_000_000:
            return f"{num / 1_000_000_000_000:,.{self.decimals}f}T"
        elif abs_num >= 1_000_000_000:
            return f"{num / 1_000_000_000:,.{self.decimals}f}B"
        elif abs_num >= 1_000_000:
            return f"{num / 1_000_000:,.{self.decimals}f}M"
        elif abs_num >= 1_000:
            return f"{num / 1_000:,.{self.decimals}f}K"
        else:
            return f"{num:,.{self.decimals}f}"

    def to_pandas_format(self) -> Union[str, Callable]:
        """Convert to a pandas Styler format specification.

        Returns
        -------
        str or callable
            A format string or callable for use with pandas Styler.format().
        """
        if self.compact:
            # Return the format_value method as a callable for compact formatting
            return self.format_value

        if self.scale_by == 100 and self.suffix == "%":
            # Use pandas' native percent formatting
            return f"{{:.{self.decimals}%}}"

        if self.suffix:
            # Need a callable for suffix
            return self.format_value

        # Simple numeric format
        return f"{{:,.{self.decimals}f}}"

    def apply_to_gt(self, gt, columns: list) -> "GT":  # type: ignore[name-defined]
        """Apply this format to a great_tables GT object.

        Parameters
        ----------
        gt : great_tables.GT
            The GT object to format.
        columns : list
            List of column names to apply formatting to.

        Returns
        -------
        great_tables.GT
            The GT object with formatting applied.
        """
        if self.scale_by == 100 and self.suffix == "%":
            # Use great_tables' native percent formatting
            return gt.fmt_percent(decimals=self.decimals, columns=columns)

        if self.compact:
            return gt.fmt_number(decimals=self.decimals, compact=True, columns=columns)

        # Use fmt_number with scale_by if needed
        if self.scale_by != 1.0:
            return gt.fmt_number(
                decimals=self.decimals, scale_by=self.scale_by, columns=columns
            )

        return gt.fmt_number(decimals=self.decimals, columns=columns)

    def to_polars_expr(self, column: str) -> pl.Expr:
        """Create a Polars expression that formats a column as a string.

        Parameters
        ----------
        column : str
            The name of the column to format.

        Returns
        -------
        pl.Expr
            A Polars expression that produces formatted string values.

        Examples
        --------
        >>> fmt = NumberFormat(decimals=2)
        >>> df = pl.DataFrame({"value": [1234.567, 89.1]})
        >>> df.with_columns(fmt.to_polars_expr("value").alias("formatted"))
        shape: (2, 2)
        ┌──────────┬───────────┐
        │ value    ┆ formatted │
        │ ---      ┆ ---       │
        │ f64      ┆ str       │
        ╞══════════╪═══════════╡
        │ 1234.567 ┆ 1,234.57  │
        │ 89.1     ┆ 89.10     │
        └──────────┴───────────┘

        >>> pct_fmt = NumberFormat(decimals=1, scale_by=100, suffix="%")
        >>> df = pl.DataFrame({"rate": [0.1234, 0.567]})
        >>> df.with_columns(pct_fmt.to_polars_expr("rate").alias("formatted"))
        shape: (2, 2)
        ┌────────┬───────────┐
        │ rate   ┆ formatted │
        │ ---    ┆ ---       │
        │ f64    ┆ str       │
        ╞════════╪═══════════╡
        │ 0.1234 ┆ 12.3%     │
        │ 0.567  ┆ 56.7%     │
        └────────┴───────────┘
        """
        # Capture instance attributes for closure
        format_func = self.format_value

        return pl.col(column).map_elements(format_func, return_dtype=pl.Utf8)

    def format_polars_column(
        self, df: pl.DataFrame, column: str, output_column: Optional[str] = None
    ) -> pl.DataFrame:
        """Format a column in a Polars DataFrame, returning a new DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            The source DataFrame.
        column : str
            The name of the column to format.
        output_column : str, optional
            The name for the formatted output column.
            If None, uses "{column}_formatted".

        Returns
        -------
        pl.DataFrame
            DataFrame with the additional formatted column.

        Examples
        --------
        >>> fmt = NumberFormat(decimals=2, scale_by=100, suffix="%")
        >>> df = pl.DataFrame({"rate": [0.1234, 0.567]})
        >>> fmt.format_polars_column(df, "rate")
        shape: (2, 2)
        ┌────────┬────────────────┐
        │ rate   ┆ rate_formatted │
        │ ---    ┆ ---            │
        │ f64    ┆ str            │
        ╞════════╪════════════════╡
        │ 0.1234 ┆ 12.34%         │
        │ 0.567  ┆ 56.70%         │
        └────────┴────────────────┘
        """
        if output_column is None:
            output_column = f"{column}_formatted"

        return df.with_columns(self.to_polars_expr(column).alias(output_column))

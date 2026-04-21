"""Polars expression and frame helpers (queries, sampling, schema, overlap)."""

import re
from collections.abc import Iterable
from typing import overload

import polars as pl

from ..types import QUERY
from ._common import F, logger
from ._dates import parse_pega_date_time_formats
from ._namespacing import _capitalize


# Pattern for validating Polars duration strings (e.g., "1d", "2w", "1h30m")
# Used by dt.truncate(), group_by_dynamic(), and other time-grouping operations
# Bounded quantifiers prevent ReDoS attacks (polynomial regex backtracking)
POLARS_DURATION_PATTERN = re.compile(r"(?:[1-9]\d{0,9}(?:ns|us|ms|s|m|h|d|w|mo|q|y)){1,10}")


def is_valid_polars_duration(value: str, max_length: int = 30) -> bool:
    """Validate Polars duration syntax.

    Checks if a string is a valid Polars duration (e.g., "1d", "1w", "1mo", "1h30m").
    Used to validate user input before passing to Polars methods like dt.truncate()
    or group_by_dynamic().

    Parameters
    ----------
    value : str
        The duration string to validate.
    max_length : int, default 30
        Maximum allowed string length (prevents excessive input).

    Returns
    -------
    bool
        True if the string is a valid Polars duration, False otherwise.

    Examples
    --------
    >>> is_valid_polars_duration("1d")
    True
    >>> is_valid_polars_duration("1w")
    True
    >>> is_valid_polars_duration("1h30m")
    True
    >>> is_valid_polars_duration("invalid")
    False
    >>> is_valid_polars_duration("")
    False

    """
    if not value or len(value) > max_length:
        return False
    return bool(POLARS_DURATION_PATTERN.fullmatch(value))


@overload
def _apply_query(
    df: pl.LazyFrame,
    query: QUERY | None = None,
    allow_empty: bool = False,
) -> pl.LazyFrame: ...


@overload
def _apply_query(
    df: pl.DataFrame,
    query: QUERY | None = None,
    allow_empty: bool = False,
) -> pl.DataFrame: ...


def _apply_query(df: F, query: QUERY | None = None, allow_empty: bool = False) -> F:
    if query is None:
        return df

    if isinstance(query, pl.Expr):
        col_names = set(query.meta.root_names())
        query = [query]
    elif isinstance(query, (list, tuple)):
        if not query:
            return df
        if not all(isinstance(expr, pl.Expr) for expr in query):
            raise ValueError(
                "If query is a list or tuple, all items need to be Expressions.",
            )
        col_names = {root_name for expr in query for root_name in expr.meta.root_names()}
    elif isinstance(query, dict):
        if not query:  # Handle empty dict
            return df
        col_names = set(query.keys())
        query = [pl.col(k).is_in(v) for k, v in query.items()]
    else:
        raise ValueError(f"Unsupported query type: {type(query)}")

    # Check if any column names were extracted
    if not col_names:
        raise ValueError("No valid column names found in the query.")

    # Check if all queried columns exist in the DataFrame
    df_columns = set(df.collect_schema().names())
    col_diff = col_names - df_columns
    if col_diff:
        raise ValueError(f"Columns not found: {col_diff}")
    filtered_df = df.filter(query)
    if not allow_empty:
        if filtered_df.lazy().select(pl.first().len()).collect().item() == 0:
            raise ValueError("The given query resulted in an empty dataframe")
    return filtered_df


def _combine_queries(existing_query: QUERY, new_query: pl.Expr) -> QUERY:
    if isinstance(existing_query, pl.Expr):
        return existing_query & new_query
    if isinstance(existing_query, list):
        return existing_query + [new_query]
    if isinstance(existing_query, dict):
        # Convert the dictionary to a list of expressions
        existing_exprs = [pl.col(k).is_in(v) for k, v in existing_query.items()]
        return existing_exprs + [new_query]
    raise ValueError("Unsupported query type")


def _polars_capitalize(df: F, extra_endwords: Iterable[str] | None = None) -> F:
    cols = df.collect_schema().names()
    renamed_cols = _capitalize(cols, extra_endwords)

    def deduplicate(columns: list[str]):
        seen: dict[str, int] = {}
        new_columns: list[str] = []
        for column in columns:
            if column not in seen:
                seen[column] = 1
            else:
                seen[column] += 1
            if seen[column] == 1:
                new_columns.append(column)
            elif (count := seen[column]) > 1:
                new_columns.append(column + f"_{count}")
            else:
                raise ValueError(f"While deduplicating:{column}")
        return new_columns

    if len(renamed_cols) != len(set(renamed_cols)):
        renamed_cols = deduplicate(renamed_cols)

    return df.rename(
        dict(
            zip(
                cols,
                renamed_cols,
                strict=False,
            ),
        ),
    )


def _extract_keys(
    df: F,
    key: str = "Name",
    capitalize: bool = True,
) -> F:
    """Extracts keys out of the pyName column

    This is not a lazy operation as we don't know the possible keys
    in advance. For that reason, we select only the key column,
    extract the keys from that, and then collect the resulting dataframe.
    This dataframe is then joined back to the original dataframe.

    This is relatively efficient, but we still do need the whole
    pyName column in memory to do this, so it won't work completely
    lazily from e.g. s3. That's why it only works with eager mode.

    The data in column for which the JSON is extract is normalized a
    little by taking out non-space, non-printable characters. Not just
    ASCII of course. This may be relatively expensive.

    JSON extraction only happens on the unique values so saves a lot
    of time with multiple snapshots of the same models, it also only
    processes rows for which the key column appears to be valid JSON.
    It will break when you "trick" it with malformed JSON.

    Column values for columns that are also encoded in the key column
    will be overwritten with values from the key column, but only for
    rows that are JSON. In previous versions all values were overwritten
    resulting in many nulls.

    Parameters
    ----------
    df: pl.DataFrame | pl.LazyFrame
        The dataframe to extract the keys from
    key: str
        The column with embedded JSON
    capitalize: bool
        If True (default) normalizes the names of the embedded columns
        otherwise keeps the names as-is.

    """
    # Checking for the 'column is None/Null' case
    if df.collect_schema()[key] != pl.Utf8:
        return df

    # Checking for the 'empty df' of 'not containing JSON' case
    if (
        len(
            df.lazy().select(key).filter(pl.col(key).str.starts_with("{")).head(1).collect(),
        )
        == 0
    ):
        return df

    keys_decoded = (
        df.filter(pl.col(key).str.starts_with("{"))
        .select(
            pl.col(key).alias("__original").unique(maintain_order=True),
        )
        .select(
            pl.col("__original"),
            pl.col("__original").cast(pl.Utf8).alias("__keys"),
            # .str.json_decode(infer_schema_length=None),
            # safe_name("__original").str.json_decode(infer_schema_length=None),
        )
        .lazy()
        .collect()
        .map_columns(
            ["__keys"],
            lambda s: s.str.json_decode(infer_schema_length=1_000_000_000),
        )
        .unnest("__keys")
        .lazy()
        .collect()
    )
    if capitalize:
        keys_decoded = _polars_capitalize(keys_decoded)

    overlap = set(df.collect_schema().names()).intersection(
        keys_decoded.collect_schema().names(),
    )
    return (
        df.join(
            keys_decoded.lazy() if isinstance(df, pl.LazyFrame) else keys_decoded,
            left_on=key,
            right_on="__original",
            coalesce=True,
            suffix="_decoded",
            how="left",
        )
        # Overwrite values from columns that also appear in the decoded keys
        .with_columns(
            [
                pl.when(pl.col(f"{c}_decoded").is_not_null()).then(pl.col(f"{c}_decoded")).otherwise(pl.col(c)).alias(c)
                for c in overlap
            ],
        )
        .drop([f"{c}_decoded" for c in overlap])
    )


def weighted_average_polars(
    vals: str | pl.Expr,
    weights: str | pl.Expr,
) -> pl.Expr:
    if isinstance(vals, str):
        vals = pl.col(vals)
    if isinstance(weights, str):
        weights = pl.col(weights)

    return (
        (vals * weights).filter(vals.is_not_nan() & vals.is_infinite().not_() & weights.is_not_null()).sum()
    ) / weights.filter(
        vals.is_not_nan() & vals.is_infinite().not_() & weights.is_not_null(),
    ).sum()


def weighted_performance_polars(
    vals: str | pl.Expr = "Performance",
    weights: str | pl.Expr = "ResponseCount",
) -> pl.Expr:
    """Polars function to return a weighted performance"""
    return weighted_average_polars(vals, weights).fill_nan(0.5)


def overlap_matrix(
    df: pl.DataFrame,
    list_col: str,
    by: str,
    show_fraction: bool = True,
) -> pl.DataFrame:
    """Calculate the overlap of a list element with all other list elements returning a full matrix.

    For each list in the specified column, this function calculates the overlap ratio (intersection size
    divided by the original list size) with every other list in the column, including itself. The result
    is a matrix where each row represents the overlap ratios for one list with all others.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the list column and grouping column.
    list_col : str
        The name of the column containing the lists. Each element in this column should be a list.
    by : str
        The name of the column to use for grouping and labeling the rows in the result matrix.

    Returns
    -------
    pl.DataFrame
        A DataFrame where:
        - Each row represents the overlap ratios for one list with all others
        - Each column (except the last) represents the overlap ratio with a specific list
        - Column names are formatted as "Overlap_{list_col_name}_{by}"
        - The last column contains the original values from the 'by' column

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "Channel": ["Mobile", "Web", "Email"],
    ...     "Actions": [
    ...         [1, 2, 3],
    ...         [2, 3, 4, 6],
    ...         [3, 5, 7, 8]
    ...     ]
    ... })
    >>> overlap_matrix(df, "Actions", "Channel")
    shape: (3, 4)
    ┌───────────────────┬───────────────┬───────────────┬─────────┐
    │ Overlap_Actions_M… │ Overlap_Actio… │ Overlap_Actio… │ Channel │
    │ ---               │ ---           │ ---           │ ---     │
    │ f64               │ f64           │ f64           │ str     │
    ╞═══════════════════╪═══════════════╪═══════════════╪═════════╡
    │ 1.0               │ 0.6666667     │ 0.3333333     │ Mobile  │
    │ 0.5               │ 1.0           │ 0.25          │ Web     │
    │ 0.25              │ 0.25          │ 1.0           │ Email   │
    └───────────────────┴───────────────┴───────────────┴─────────┘

    """
    n = df.height
    by_values = df[by].to_list()

    if n == 0:
        return pl.DataFrame().with_columns(pl.Series(df[by]))

    list_expr = pl.col(list_col)
    if df.schema[list_col] == pl.List(pl.Null):
        list_expr = list_expr.cast(pl.List(pl.Int64))

    df_sets = df.lazy().select(list_expr.list.unique().alias("__L")).with_row_index("__idx")

    left = df_sets.rename({"__L": "__L_j", "__idx": "__j"})
    right = df_sets.rename({"__L": "__L_i", "__idx": "__i"})

    pairs = left.join(right, how="cross").with_columns(
        pl.col("__L_i").list.set_intersection(pl.col("__L_j")).list.len().alias("__isect"),
        pl.col("__L_j").list.len().alias("__len_j"),
    )
    if show_fraction:
        pairs = pairs.with_columns(
            pl.when(pl.col("__i") == pl.col("__j"))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("__isect").cast(pl.Float64) / pl.col("__len_j").cast(pl.Float64))
            .alias("__val"),
        )
    else:
        pairs = pairs.with_columns(pl.col("__isect").cast(pl.Int64).alias("__val"))

    pairs_df = pairs.sort("__j", "__i").collect()
    wide = pairs_df.pivot(values="__val", index="__j", on="__i", sort_columns=False).sort("__j").drop("__j")

    cur_cols = wide.columns
    rename_map = {cur_cols[i]: f"Overlap_{list_col}_{by_values[i]}" for i in range(n)}
    wide = wide.rename(rename_map)
    return wide.with_columns(pl.Series(df[by]))


def overlap_lists_polars(col: pl.Series) -> pl.Series:
    """Calculate the average overlap ratio of each list element with all other list elements into a single Series.

    For each list in the input Series, this function calculates the average overlap (intersection)
    with all other lists, normalized by the size of the original list. The overlap ratio represents
    how much each list has in common with all other lists on average.

    Parameters
    ----------
    col : pl.Series
        A Polars Series where each element is a list. The function will calculate
        the overlap between each list and all other lists in the Series.

    Returns
    -------
    pl.Series
        A Polars Series of float values representing the average overlap ratio for each list.
        Each value is calculated as:
        (sum of intersection sizes with all other lists) / (number of other lists) / (size of original list)

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.Series([
    ...     [1, 2, 3],
    ...     [2, 3, 4, 6],
    ...     [3, 5, 7, 8]
    ... ])
    >>> overlap_lists_polars(data)
    shape: (3,)
    Series: '' [f64]
    [
        0.5
        0.375
        0.25
    ]
    >>> df = pl.DataFrame({"Channel" : ["Mobile", "Web", "Email"], "Actions" : pl.Series([
    ...     [1, 2, 3],
    ...     [2, 3, 4, 6],
    ...     [3, 5, 7, 8]
    ... ])})
    >>> df.with_columns(pl.col("Actions").map_batches(overlap_lists_polars))
    shape: (3, 2)
    ┌─────────┬─────────┐
    │ Channel │ Actions │
    │ ---     │ ---     │
    │ str     │ f64     │
    ╞═════════╪═════════╡
    │ Mobile  │ 0.5     │
    │ Web     │ 0.375   │
    │ Email   │ 0.25    │
    └─────────┴─────────┘

    """
    n = col.len()
    if n == 0:
        return pl.Series([], dtype=pl.Float64)
    if n == 1:
        return pl.Series([0.0], dtype=pl.Float64)

    col_for_lazy = col
    if col.dtype == pl.List(pl.Null):
        col_for_lazy = col.cast(pl.List(pl.Int64))

    df_sets = (
        pl.DataFrame({"__L": col_for_lazy})
        .lazy()
        .select(pl.col("__L").list.unique().alias("__L"))
        .with_columns(pl.col("__L").list.len().alias("__len"))
        .with_row_index("__i")
    )

    left = df_sets.rename({"__L": "__L_i", "__len": "__len_i"})
    right = df_sets.select(pl.col("__L").alias("__L_j"), pl.col("__i").alias("__j"))

    pairs = (
        left.join(right, how="cross")
        .filter(pl.col("__i") != pl.col("__j"))
        .with_columns(
            pl.col("__L_i").list.set_intersection(pl.col("__L_j")).list.len().alias("__isect"),
        )
    )

    agg = (
        pairs.group_by("__i", maintain_order=False)
        .agg(
            pl.sum("__isect").alias("__sum"),
            pl.first("__len_i").alias("__len_i"),
        )
        .sort("__i")
        .with_columns(
            pl.when(pl.col("__len_i") == 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("__sum").cast(pl.Float64) / (n - 1) / pl.col("__len_i"))
            .alias("__avg"),
        )
        .select("__avg")
        .collect()
    )

    return agg["__avg"].rename("")


def lazy_sample(df: F, n_rows: int, with_replacement: bool = True) -> F:
    if with_replacement:
        return df.select(pl.all().sample(n=n_rows, with_replacement=with_replacement))

    from functools import partial

    import numpy as np

    def sample_it(s: pl.Series, n) -> pl.Series:
        s_len = s.len()
        if s_len < n:
            return pl.Series(values=[True] * s_len, dtype=pl.Boolean)
        return pl.Series(
            values=np.random.binomial(1, n / s_len, s_len),
            dtype=pl.Boolean,
        )

    func = partial(sample_it, n=n_rows)
    return df.with_columns(pl.first().map_batches(func).alias("_sample")).filter(pl.col("_sample")).drop("_sample")


def _apply_schema_types(df: F, definition, **timestamp_opts) -> F:
    """This function is used to convert the data types of columns in a DataFrame to a desired types.
    The desired types are defined in a `PegaDefaultTables` class.

    Parameters
    ----------
    df : pl.LazyFrame
        The DataFrame whose columns' data types need to be converted.
    definition : PegaDefaultTables
        A `PegaDefaultTables` object that contains the desired data types for the columns.
    timestamp_opts : str
        Additional arguments for timestamp parsing.

    Returns
    -------
    list
        A list with polars expressions for casting data types.

    """

    def get_mapping(columns, reverse=False):
        if not reverse:
            return dict(zip(columns, _capitalize(columns), strict=False))
        return dict(zip(_capitalize(columns), columns, strict=False))

    schema = df.collect_schema()
    named = get_mapping(schema.names())
    typed = get_mapping(
        [col for col in dir(definition) if not col.startswith("__")],
        reverse=True,
    )

    types = []
    for col, renamedCol in named.items():
        try:
            new_type = getattr(definition, typed[renamedCol])
            original_type = schema[col].base_type()
            if original_type == pl.Null:
                logger.debug("Column %s has Null data type; skipping cast.", col)
            elif original_type != new_type:
                if original_type == pl.Categorical and new_type.is_numeric():
                    types.append(pl.col(col).cast(pl.Utf8).cast(new_type))
                elif new_type == pl.Datetime and original_type != pl.Date:
                    types.append(parse_pega_date_time_formats(col, **timestamp_opts))
                else:
                    types.append(pl.col(col).cast(new_type, strict=False))
        except (KeyError, AttributeError) as exc:
            logger.debug(
                "Column %s not in default table schema; can't set type: %s",
                col,
                exc,
            )
    return df.with_columns(types)

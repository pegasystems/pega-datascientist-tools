"""Small Polars helpers and aggregations used by Quarto reports."""

import polars as pl


def polars_col_exists(df, col):
    return col in df.collect_schema().names() and df.collect_schema()[col] != pl.Null


def polars_subset_to_existing_cols(all_columns, cols):
    return [col for col in cols if col in all_columns]


def n_unique_values(dm, all_dm_cols, fld):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return 0
    return dm.model_data.select(pl.col(fld)).drop_nulls().collect().n_unique()


def max_by_hierarchy(dm, all_dm_cols, fld, grouping):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return 0
    grouping = polars_subset_to_existing_cols(all_dm_cols, grouping)
    if len(grouping) == 0:
        return 0
    return (
        dm.model_data.group_by(grouping)
        .agg(pl.col(fld).drop_nulls().n_unique())
        .select(fld)
        .drop_nulls()
        .max()
        .collect()
        .item()
    )


def avg_by_hierarchy(dm, all_dm_cols, fld, grouping):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return 0
    grouping = polars_subset_to_existing_cols(all_dm_cols, grouping)
    if len(grouping) == 0:
        return 0
    return (
        dm.model_data.group_by(grouping)
        .agg(pl.col(fld).drop_nulls().n_unique())
        .select(fld)
        .drop_nulls()
        .mean()
        .collect()
        .item()
    )


def sample_values(dm, all_dm_cols, fld, n=6):
    if not isinstance(fld, list):
        fld = [fld]
    fld = polars_subset_to_existing_cols(all_dm_cols, fld)
    if len(fld) == 0:
        return "-"
    return (
        dm.model_data.select(
            pl.concat_str(fld, separator="/").alias("__SampleValues__"),
        )
        .drop_nulls()
        .collect()
        .to_series()
        .unique()
        .sort()
        .to_list()[:n]
    )


def gains_table(
    df: pl.LazyFrame | pl.DataFrame,
    value: str,
    index: str | None = None,
    by: str | list[str] | None = None,
) -> pl.DataFrame:
    """Calculate cumulative gains for visualization.

    Computes cumulative distribution of a value metric, sorted by the ratio
    of value to index (or by value alone if no index). Used for gains charts
    to show model response skewness.

    Parameters
    ----------
    df : pl.LazyFrame | pl.DataFrame
        Input data containing the value and optional index columns
    value : str
        Column name containing the metric to compute gains for (e.g., "ResponseCount")
    index : str, optional
        Column name to normalize by (e.g., population size). If None, uses row count.
    by : str | list[str], optional
        Column(s) to group by for separate gain curves. If None, computes single curve.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - cum_x: Cumulative proportion of index (or models)
        - cum_y: Cumulative proportion of value
        - by columns: if `by` is specified

    Examples
    --------
    >>> # Single gains curve for response count
    >>> gains = gains_table(df, value="ResponseCount")

    >>> # Gains curves by channel, normalized by population
    >>> gains = gains_table(df, value="Positives", index="Population", by="Channel")
    """
    # Determine sorting expression
    sort_expr = pl.col(value) if index is None else pl.col(value) / pl.col(index)

    # Determine index expression for cumulative x-axis
    if index is None:
        index_expr = pl.int_range(1, pl.len() + 1) / pl.len()
    else:
        index_expr = pl.cum_sum(index) / pl.sum(index)

    if by is None:
        # Single gains curve
        gains_df = pl.concat(
            [
                pl.DataFrame(data={"cum_x": [0.0], "cum_y": [0.0]}).lazy(),
                df.lazy()
                .sort(sort_expr, descending=True)
                .select(
                    index_expr.cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value)).cast(pl.Float64).alias("cum_y"),
                ),
            ]
        )
    else:
        # Multiple gains curves grouped by column(s)
        by_as_list = by if isinstance(by, list) else [by]
        sort_expr_with_by = by_as_list + [sort_expr]
        gains_df = (
            df.lazy()
            .sort(sort_expr_with_by, descending=True)
            .select(
                by_as_list
                + [
                    index_expr.over(by).cast(pl.Float64).alias("cum_x"),
                    (pl.cum_sum(value) / pl.sum(value)).over(by).cast(pl.Float64).alias("cum_y"),
                ]
            )
        )
        # Add entry for the (0,0) point for each group
        gains_df = pl.concat([gains_df.group_by(by).agg(cum_x=pl.lit(0.0), cum_y=pl.lit(0.0)), gains_df]).sort(
            by_as_list + ["cum_x"]
        )

    return gains_df.collect()

"""Expected schema of the pre-aggregated explanations parquet files."""

from __future__ import annotations

import polars as pl

AGGREGATE_SCHEMA: dict[str, pl.DataType] = {
    "context_partition": pl.String(),
    "predictor_name": pl.String(),
    "predictor_type": pl.String(),
    "bin_contents": pl.String(),
    "bin_order": pl.Int64(),
    "contribution": pl.Float64(),
    "contribution_abs": pl.Float64(),
    "contribution_min": pl.Float64(),
    "contribution_max": pl.Float64(),
    "frequency": pl.Int64(),
}
"""Columns read from ``OVERVIEW.parquet`` / ``BY_CONTEXT.parquet``, and the
dtype each is cast to. Casting on read means every downstream aggregation
sees the same types regardless of how the exporting Pega version wrote the
file."""


def apply_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Select the expected columns from *lf* and cast them to their dtypes.

    Parameters
    ----------
    lf : pl.LazyFrame
        Scan over one of the aggregated parquet files.

    Returns
    -------
    pl.LazyFrame
        *lf* narrowed to :data:`AGGREGATE_SCHEMA` and cast to its dtypes.

    Raises
    ------
    ValueError
        If *lf* is missing any of the expected columns.

    """
    missing = [col for col in AGGREGATE_SCHEMA if col not in lf.collect_schema().names()]
    if missing:
        raise ValueError(f"Aggregated data is missing expected column(s): {', '.join(missing)}")
    return lf.select(pl.col(col).cast(dtype) for col, dtype in AGGREGATE_SCHEMA.items())

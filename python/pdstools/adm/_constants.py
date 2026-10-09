"""Shared ADM constants."""

import polars as pl

MIN_CHANNEL_POSITIVES = 200
"""Minimum total positives for a channel to have sufficient feedback."""

MIN_CHANNEL_RESPONSES = 1_000
"""Minimum total responses for a channel to have sufficient feedback."""


def channel_is_valid_expr() -> pl.Expr:
    """Expression flagging channels with sufficient feedback.

    Expects ``TotalPositives`` and ``TotalResponseCount`` columns.
    """
    return (pl.col("TotalPositives") >= MIN_CHANNEL_POSITIVES) & (pl.col("TotalResponseCount") >= MIN_CHANNEL_RESPONSES)

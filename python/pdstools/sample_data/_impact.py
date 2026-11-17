"""Impact Analyzer view: daily experiment counts aggregated from sampled IH exposures."""

from __future__ import annotations

import polars as pl

from ._ih import IMPRESSION_OUTCOME, POSITIVE_OUTCOME

_DIMENSIONS = ("SnapshotTime", "ControlGroup", "Channel", "Issue", "Group", "Name", "Treatment")


def impact_analyzer_data(interactions: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate raw sample IH rows to Impact Analyzer's long format.

    Each exposure counts once: an impression when it was offered, an accept
    when it reached the channel's positive outcome, and its accepted value
    once. Exposures are attributed to the day they were offered, so the
    counts reconcile exactly with the IH rows they come from.

    Parameters
    ----------
    interactions : pl.LazyFrame
        Raw rows from :func:`pdstools.sample_data._ih.interaction_history`.

    Returns
    -------
    pl.LazyFrame
        One row per day, control group, ``Channel/Direction`` and treatment,
        with ``Impressions``, ``Accepts`` and ``ValuePerImpression``.
    """
    impression = pl.col("pyOutcome") == pl.col("pyChannel").replace_strict(IMPRESSION_OUTCOME)
    accept = pl.col("pyOutcome") == pl.col("pyChannel").replace_strict(POSITIVE_OUTCOME)
    exposures = interactions.group_by("ExposureID").agg(
        pl.col("pxOutcomeTime").filter(impression).min().dt.truncate("1d").alias("SnapshotTime"),
        pl.col("MktValue").first().str.strip_prefix("NBAHealth_").alias("ControlGroup"),
        pl.concat_str("pyChannel", "pyDirection", separator="/").first().alias("Channel"),
        pl.col("pyIssue").first().alias("Issue"),
        pl.col("pyGroup").first().alias("Group"),
        pl.col("pyName").first().alias("Name"),
        pl.col("pyTreatment").first().alias("Treatment"),
        impression.any().alias("Impressed"),
        accept.any().alias("Accepted"),
        pl.col("Value").filter(accept).first().fill_null(0.0).alias("Value"),
    )
    return (
        exposures.filter(pl.col("Impressed"))
        .group_by(_DIMENSIONS)
        .agg(
            pl.len().cast(pl.Float64).alias("Impressions"),
            pl.col("Accepted").sum().cast(pl.Float64).alias("Accepts"),
            (pl.col("Value").sum() / pl.len()).alias("ValuePerImpression"),
        )
        .sort(_DIMENSIONS)
    )

from functools import cache, cached_property
from typing import TYPE_CHECKING, Dict, Optional

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from .ValueFinder import ValueFinder


class Aggregates:
    def __init__(self, vf: "ValueFinder"):
        self.vf = vf
        self._quantile_from_threshold: Dict[float, float] = {}

    def get_customer_summary(
        self,
        *,
        threshold: Optional[float] = None,
    ) -> pl.LazyFrame:
        """Computes the summary of propensities for all customers

        Parameters
        ----------
        th: Optional[float]
            The threshold to consider an action 'good'.
            If a customer has actions with propensity above this,
            the customer has at least one relevant action.
            If not given, will default to 5th quantile.
        """
        threshold = threshold or self.vf.threshold
        return (
            self.vf.df.group_by("CustomerID", "Stage")
            .agg(
                MaxPropensity=pl.max("Propensity"),
                MaxFinalPropensity=pl.max("FinalPropensity"),
                MaxModelPropensity=pl.max("ModelPropensity"),
                NOffers=pl.count("Propensity"),
            )
            .with_columns(
                RelevantActions=pl.col("MaxModelPropensity") >= pl.lit(threshold),
                IrrelevantActions=pl.col("MaxModelPropensity") < pl.lit(threshold),
            )
        )

    def get_counts_per_stage(self, *, threshold: Optional[float] = None):
        threshold = threshold or self.vf.threshold
        customer_summary = self.get_customer_summary(threshold=threshold)
        return (
            customer_summary.group_by("Stage")
            .agg(
                pl.sum("RelevantActions"),
                pl.sum("IrrelevantActions"),
                pl.count("RelevantActions").alias("NoActions"),
            )
            .with_columns(NoActions=pl.lit(self.vf.n_customers) - pl.col("NoActions"))
        )

    @cached_property
    def max_propensity_per_customer(self) -> pl.DataFrame:
        return (
            self.vf.df.group_by(["CustomerID", "Stage"])
            .agg(pl.max("ModelPropensity"))
            .collect()
        )

    @cache
    def get_threshold_from_quantile(self, quantile: float) -> float:
        threshold = (
            self.vf.df.filter(pl.col("Stage") == "Eligibility")
            .select(pl.col("ModelPropensity").quantile(quantile))
            .collect()
            .item()
        )

        # maintained for reverse mapping.
        # private attribute because it's only populated on cache miss,
        # that might not be intuitive behavior for public usage
        self._quantile_from_threshold[threshold] = quantile
        return threshold

    @cache
    def get_counts_for_threshold(self, threshold: float) -> pl.DataFrame:
        return (
            self.max_propensity_per_customer.with_columns(
                RelevantActions=pl.col("ModelPropensity") >= pl.lit(threshold),
                IrrelevantActions=pl.col("ModelPropensity") < pl.lit(threshold),
            )
            .group_by("Stage")
            .agg(
                pl.sum("RelevantActions"),
                pl.sum("IrrelevantActions"),
                pl.count("RelevantActions").alias("NoActions"),
            )
            .with_columns(NoActions=pl.lit(self.vf.n_customers) - pl.col("NoActions"))
        )

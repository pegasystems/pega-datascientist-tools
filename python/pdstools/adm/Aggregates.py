from typing import TYPE_CHECKING, Literal, Optional

import polars as pl

from ..utils import cdh_utils
from ..utils.types import QUERY

if TYPE_CHECKING:
    from .new_ADMDatamart import ADMDatamart


class Aggregates:
    def __init__(self, datamart: "ADMDatamart"):
        self.datamart = datamart

    def last(
        self,
        *,
        data: Optional[pl.LazyFrame] = None,
        table: Literal["model_data", "predictor_data", "combined_data"] = "model_data",
    ):
        """Gets the last snapshot of the given table

        Parameters
        ----------
        data : Optional[pl.LazyFrame], optional
            If provided, subsets to just that dataframe, by default None
        table : Literal['model_data', 'predictor_data', 'combined_data'], optional
            If provided, specifies the table to get data from, by default "model_data"

        Returns
        -------
        _type_
            _description_
        """
        df: pl.LazyFrame = data if data is not None else getattr(self.datamart, table)
        if df.collect_schema()["SnapshotTime"] == pl.Null:
            return df

        return df.filter(
            pl.col("SnapshotTime").fill_null(strategy="zero")
            == pl.col("SnapshotTime").fill_null(strategy="zero").max()
        )

    def _combine_data(
        self, model_df: Optional[pl.LazyFrame], predictor_df: Optional[pl.LazyFrame]
    ) -> Optional[pl.LazyFrame]:
        """Combines the model and predictor tables to the `combined_data` attribute

        Parameters
        ----------
        model_df : pl.LazyFrame
            The model snapshots table
        predictor_df : pl.LazyFrame
            The predictor binning snapshots table

        Returns
        -------
        pl.LazyFrame
            The resulting data, joined on the ModelID column
        """
        if model_df is None or predictor_df is None:
            return None
        return self.last(data=model_df).join(
            self.last(data=predictor_df), on="ModelID", suffix="Bin"
        )

    def predictor_performance_pivot(
        self,
        *,
        query: Optional[QUERY] = None,
        active_only: bool = False,
        by="Name",
        top_predictors: Optional[int] = None,
        top_groups: Optional[int] = None,
    ) -> pl.LazyFrame:
        """Creates a pivot table of the predictor performance per 'group'

        Parameters
        ----------
        query : Optional[QUERY], optional
            A query to apply to the data before creating the pivot, by default None
        by : str, optional
            A group by which to 'facet', by default "Name".
            If, for instance, the 'by' argument is set to 'Configuration',
            each row will be a distinct configuration
        top_predictors : Optional[int], optional
            Specify the maximum number of predictors, by default None
        top_groups : Optional[int], optional
            Specify the maximum number of 'groups'
            specified in the 'by' argument, by default None

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with a column for each predictor, and a row for each 'group'.
            The values represent the weighted performance for that predictor
        """
        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data").filter(
                pl.col("PredictorName") != "Classifier"
            ),
            query,
        )
        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")
        unique_predictors = df.select(pl.col("PredictorName").unique()).collect()[
            "PredictorName"
        ]
        q = (
            (
                df.filter(pl.col("ResponseCount") > 0)
                .with_columns(
                    pl.col("PerformanceBin").fill_nan(0.5),  # should we do this?
                )
                .unique(subset=[by, "PredictorName"], keep="first")
                .group_by(by, "PredictorName")
                .agg(
                    cdh_utils.weighted_average_polars(
                        "PerformanceBin", "ResponseCountBin"
                    )
                )
            )
            .group_by(by)
            .agg(
                [
                    (
                        pl.when(pl.col("PredictorName") == predictor)
                        .then(pl.col("PerformanceBin"))
                        .otherwise(pl.lit(0.5))
                        .alias(predictor)
                    )
                    for predictor in unique_predictors
                ]
            )
            .with_columns(pl.all().exclude(by).list.max())
        ).sort(pl.mean_horizontal(pl.all().exclude(by)), descending=True)

        column_order = (
            q.select(pl.all().exclude(by).mean())
            .collect()
            .transpose(include_header=True)
        ).sort("column_0", descending=True)["column"]

        if top_predictors:
            column_order = column_order.head(top_predictors)
        if top_groups:
            q = q.head(top_groups)

        return q.select(by, *column_order)

    def model_summary(
        self, by: str = "Name", query: Optional[QUERY] = None
    ) -> pl.LazyFrame:
        """Generate a summary of statistic for each model (based on model ID)

        If you want to generate statistics at a model name or treatment level,
        specify this in the 'by' column.

        Parameters
        ----------
        by : str, optional
            The column to define the 'counts' for, by default "ModelID"
            Must be part of the context keys in the ADMDatamart class
        query : Optional[QUERY], optional
            A query to apply to the data before summarization, by default None

        Returns
        -------
        pl.LazyFrame
            A LazyFrame, with one row for each context key combination
        """
        df = cdh_utils._apply_query(self.datamart.aggregates.last(), query)
        aggregate_columns = ["ResponseCount", "Performance", "SuccessRate", "Positives"]

        if by != "ModelID" and by not in self.datamart.context_keys:
            raise ValueError("The 'by' column specified should be a context key.")

        group_by = (
            self.datamart.context_keys[: self.datamart.context_keys.index(by)]
            if by != "ModelID"
            else by
        )

        return (
            df.group_by(group_by)
            .agg(
                pl.count().alias("count"),
                (pl.col("ResponseCount") == 0).sum().alias("Count_without_responses"),
                pl.col("ResponseCount", "Positives").sum().name.suffix("_sum"),
                pl.col(aggregate_columns).max().name.suffix("_max"),
                pl.col(aggregate_columns).mean().name.suffix("_mean"),
                Weighted_performance=cdh_utils.weighted_performance_polars(),
                Weighted_success_rate=cdh_utils.weighted_average_polars(
                    "SuccessRate", "ResponseCount"
                ).fill_nan(0.0),
            )
            .with_columns(
                Percentage_without_responses=(
                    pl.col("Count_without_responses") / pl.col("count")
                ).fill_nan(0.0)
            )
        )

    def predictor_counts(self, *, by: str = "Type", query: Optional[QUERY] = None):
        """Returns the count of each predictor grouped by a certain column

        Parameters
        ----------
        by : str, optional
            The column to group the data by, by default "Type"
        query : Optional[QUERY], optional
            A query to apply to the data, by default None

        Returns
        -------
        pl.LazyFrame
            A LazyFrame, with one row per predictor and 'by' combo
        """
        df = (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="combined_data"), query=query
            )
            .select("Name", "EntryType", "PredictorName", by)
            .filter(pl.col("PredictorName") != "Classifier")
            .group_by(pl.all().exclude("PredictorName"))
            .agg(PredictorCount=pl.n_unique("PredictorName"))
        )
        overall = (
            df.group_by(pl.all().exclude(["PredictorName", by, "PredictorCount"]))
            .agg(pl.sum("PredictorCount"))
            .with_columns(pl.lit("Overall").alias(by))
        )

        return (
            pl.concat([df, overall.select(df.columns)])
            .with_columns(pl.col("PredictorCount").cast(pl.Int64))
            .sort(["Name", "EntryType", by])
        )

    @staticmethod
    def _top_n(
        df: pl.DataFrame,
        top_n: int,
        metric: str = "PerformanceBin",
        facets: Optional[list] = None,
    ):
        """Subsets DataFrame to contain only top_n predictors.

        Parameters
        ----------
        df : pl.DataFrame
            Table to subset
        top_n : int
            Number of top predictors
        metric: str
            Metric to use for comparing predictors
        facets : list
            Subsets top_n predictors over facets. Seperate top predictors for each facet

        Returns
        -------
        pl.DataFrame
            Subsetted dataframe
        """

        if top_n < 1:
            return df
        if facets:
            return df.join(
                df.group_by(facets + ["PredictorName"])
                .agg(cdh_utils.weighted_average_polars(metric, "ResponseCountBin"))
                .filter(pl.col(metric).is_not_nan())
                .group_by(*facets)
                .agg(
                    pl.col("PredictorName").sort_by(metric, descending=True).head(top_n)
                )
                .explode("PredictorName"),
                on=(*facets, "PredictorName"),
            )

        return df.join(
            df.group_by("PredictorName")
            .agg(cdh_utils.weighted_average_polars(metric, "ResponseCountBin"))
            .filter(pl.col(metric).is_not_nan())
            .sort(metric, descending=True)
            .head(top_n)
            .select("PredictorName"),
            on="PredictorName",
        )

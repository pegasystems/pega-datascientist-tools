from typing import TYPE_CHECKING, Dict, Literal, Optional

import polars as pl

from ..utils import cdh_utils
from ..utils.types import QUERY
from .CDH_Guidelines import CDHGuidelines

if TYPE_CHECKING:  # pragma: no cover
    from .ADMDatamart import ADMDatamart


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
        if data is None and not hasattr(self.datamart, table):
            raise ValueError(f"{table} not available in the datamart")

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

    def summary_by_channel(
        self,
        custom_channels: Optional[Dict[str, str]] = None,
        by_period: Optional[str] = None,
        keep_lists: bool = False,
    ) -> pl.LazyFrame:
        """Summarize ADM models per channel

        Parameters
        ----------
        custom_channels : Dict[str, str], optional
            Optional list with custom channel/direction name mappings. Defaults to None.
        by_period : str, optional
            Optional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. If provided, creates a new Period column with the truncated date/time. Defaults to None.
        keep_lists : bool, optional
            Internal flag to keep some columns (action and treatment names etc) as full lists.

        Returns
        -------
        pl.LazyFrame
            Dataframe with summary per channel (and optionally a period)
        """
        if self.datamart.model_data is None:
            raise ValueError("Summary by channel needs model data")

        if not custom_channels:
            custom_channels = {}

        if by_period is not None:
            period_expr = [
                pl.col("SnapshotTime")
                .dt.truncate(by_period)
                .cast(pl.Date)
                .alias("Period")
            ]
        else:
            period_expr = []
        cdh_guidelines = CDHGuidelines()

        # Removes whitespace and capitalizes names for matching
        def name_normalizer(x):
            return pl.col(x).str.replace_all(r"[ \-_]", "").str.to_uppercase()

        directionMapping = pl.DataFrame(
            # Standard directions have a 1:1 mapping to channel groups
            {
                "Direction": cdh_guidelines.standard_directions,
                "DirectionGroup": cdh_guidelines.standard_directions,
            }
        ).with_columns(normalizedDirection=name_normalizer("Direction"))

        channelGroupMapping = (
            pl.concat(
                [
                    pl.DataFrame(
                        # Standard channels have a 1:1 mapping to channel groups
                        {
                            "Channel": cdh_guidelines.standard_channels,
                            "ChannelGroup": cdh_guidelines.standard_channels,
                        }
                    ),
                    pl.DataFrame(
                        # feels like a convoluted way to put a dict into a polars dataframe
                        # but that is just what this does in the end
                        {
                            "Channel": custom_channels.keys(),
                            "ChannelGroup": [
                                custom_channels[c] for c in custom_channels.keys()
                            ],
                        }
                    ),
                ],
                how="diagonal",
            )
            .with_columns(normalizedChannel=name_normalizer("Channel"))
            .unique()
            .sort(["ChannelGroup", "Channel"])
        )

        actionIdentifierExpr = pl.concat_str(["Issue", "Group", "Name"], separator="/")
        activeActionExpr = (pl.col("ResponseCount").sum() > 0).over(
            ["Issue", "Group", "Name"]
        )

        # all these expressions needed because not every customer has Treatments and
        # polars can't aggregate literals, so we have to be careful to pass on explicit
        # values when there are no treatments
        columns = self.datamart.model_data.collect_schema().names()
        treatmentIdentifierExpr = (
            pl.concat_str(["Issue", "Group", "Name", "Treatment"], separator="/")
            if "Treatment" in columns
            else pl.lit("")
        )
        activeTreatmentExpr = (
            (
                (pl.col("ResponseCount").sum() > 0)
                & (pl.col("Treatment").is_not_null())
            ).over(["Issue", "Group", "Name", "Treatment"])
            if "Treatment" in columns
            else pl.lit(False)
        )
        uniqueTreatmentExpr = (
            treatmentIdentifierExpr.unique() if "Treatment" in columns else pl.lit([])
        )
        uniqueTreatmentCountExpr = (
            treatmentIdentifierExpr.n_unique() if "Treatment" in columns else pl.lit(0)
        )
        uniqueUsedTreatmentExpr = (
            treatmentIdentifierExpr.filter(pl.col("isUsedTreatment")).unique()
            if "Treatment" in columns
            else pl.lit([])
        )
        uniqueUsedTreatmentCountExpr = (
            treatmentIdentifierExpr.filter(pl.col("isUsedTreatment")).n_unique()
            if "Treatment" in columns
            else pl.lit(0)
        )
        channel_summary = (
            self.datamart.model_data.with_columns(
                [
                    activeActionExpr.alias("isUsedAction"),
                    activeTreatmentExpr.alias("isUsedTreatment"),
                ]
                + period_expr
            )
            # .filter(
            #     pl.col("Configuration").cast(pl.Utf8).str.to_uppercase()
            #     # TODO maybe not here...
            #     != "OMNIADAPTIVEMODEL"
            # )
            # .with_columns(
            #     OriginalChannelDirection=pl.concat_str(
            #         ["Channel", "Direction"], separator="/"
            #     )
            # )
            .with_columns(
                normalizedChannel=name_normalizer("Channel"),
                normalizedDirection=name_normalizer("Direction"),
            )
            .join(
                channelGroupMapping.lazy(),
                on="normalizedChannel",
                how="left",
                # suffix="_standard",
            )
            .join(
                directionMapping.lazy(),
                on="normalizedDirection",
                how="left",
                # suffix="_standard",
            )
            .with_columns(
                ChannelDirectionGroup=pl.when(
                    pl.col("ChannelGroup").is_not_null()
                    & pl.col("DirectionGroup").is_not_null()
                )
                .then(pl.concat_str(["ChannelGroup", "DirectionGroup"], separator="/"))
                .otherwise(pl.lit("Other")),
            )
            .group_by(
                [
                    "Channel",
                    "Direction",
                    "ChannelDirectionGroup",
                ]
                + (["Period"] if by_period is not None else [])
            )
            .agg(
                pl.col("SnapshotTime").min().cast(pl.Date).alias("DateRange Min"),
                pl.col("SnapshotTime").max().cast(pl.Date).alias("DateRange Max"),
                # pl.col("OriginalChannelDirection"),
                pl.col("Positives").sum(),
                pl.col("ResponseCount").sum(),
                (cdh_utils.weighted_performance_polars() * 100).alias("Performance"),
                pl.col("Configuration").cast(pl.Utf8),
                pl.col("Configuration")
                .cast(pl.Utf8)
                .str.to_uppercase()
                .is_in([x.upper() for x in cdh_guidelines.standard_configurations])
                .alias("isNBADModelConfiguration"),
                actionIdentifierExpr.n_unique().alias("Total Number of Actions"),
                uniqueTreatmentCountExpr.alias("Total Number of Treatments"),
                # TODO use last update property instead
                (actionIdentifierExpr.filter(pl.col("isUsedAction")).n_unique()).alias(
                    "Used Actions"
                ),
                uniqueUsedTreatmentCountExpr.alias("Used Treatments"),
                # keep lists of unique values for aggregation over channels
                AllIssues=pl.col("Issue").unique(),
                AllGroups=pl.concat_str(["Issue", "Group"], separator="/").unique(),
                AllActions=actionIdentifierExpr.unique(),
                AllTreatments=uniqueTreatmentExpr,
                AllUsedActions=actionIdentifierExpr.filter(
                    pl.col("isUsedAction")
                ).unique(),
                AllUsedTreatments=uniqueUsedTreatmentExpr,
            )
            .with_columns(
                pl.when(pl.col("Used Actions").is_not_null())
                .then(pl.col("Used Actions"))
                .otherwise(pl.lit(0))
                .alias("Used Actions"),
                pl.when(pl.col("Used Treatments").is_not_null())
                .then(pl.col("Used Treatments"))
                .otherwise(pl.lit(0))
                .alias("Used Treatments"),
                ChannelDirection=pl.format(
                    "{}/{}",
                    pl.when(pl.col("Channel").is_not_null() & (pl.col("Channel") != ""))
                    .then(pl.col("Channel"))
                    .otherwise(pl.lit("")),
                    pl.when(
                        pl.col("Direction").is_not_null() & (pl.col("Direction") != "")
                    )
                    .then(pl.col("Direction"))
                    .otherwise(pl.lit("")),
                ),
                isValid=(pl.col("Positives") > 200) & (pl.col("ResponseCount") > 1000),
                Configuration=pl.col("Configuration")
                .list.unique()
                .list.sort()
                .list.join(", "),
                Issues=pl.col("AllIssues").list.len(),
                Groups=pl.col("AllGroups").list.len(),
                # TODO: NBAD detection is not entirely correct: if the only standard model is omniadaptive, we
                # miss it, because that one got filtered out early on
                usesNBAD=pl.col("isNBADModelConfiguration").list.any(),
                usesNBADOnly=pl.col("isNBADModelConfiguration").list.any()
                & pl.col("isNBADModelConfiguration").list.all(),
            )
            .sort(
                [
                    # NB channel direction group isn't unique per se so make sure to have a fully defined order
                    "ChannelDirectionGroup",
                    "Channel",
                    "Direction",
                ]
                + (["Period"] if by_period is not None else [])
            )
        )

        item_overlap_actions = channel_summary.select(
            ["AllActions", "isValid"]
        ).collect()

        return channel_summary.with_columns(
            pl.Series(
                cdh_utils.overlap_lists_polars(
                    item_overlap_actions["AllActions"],
                    item_overlap_actions["isValid"],
                )
            ).alias("OmniChannel Actions"),
            CTR=(pl.col("Positives")) / (pl.col("ResponseCount")),
        ).drop(
            ["isNBADModelConfiguration"]
            + (
                []
                if keep_lists
                else [
                    "AllIssues",
                    "AllGroups",
                    "AllActions",
                    "AllUsedActions",
                    "AllTreatments",
                    "AllUsedTreatments",
                ]
            )
        )

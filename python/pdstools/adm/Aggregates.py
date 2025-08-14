__all__ = ["Aggregates"]
import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import polars as pl
import polars.selectors as cs

from ..utils import cdh_utils
from ..utils.types import QUERY
from .CDH_Guidelines import CDHGuidelines

if TYPE_CHECKING:  # pragma: no cover
    from .ADMDatamart import ADMDatamart


class Aggregates:
    def __init__(self, datamart: "ADMDatamart"):
        self.datamart = datamart
        self.cdh_guidelines = CDHGuidelines()

    def last(
        self,
        *,
        data: Optional[pl.LazyFrame] = None,
        table: Literal["model_data", "predictor_data", "combined_data"] = "model_data",
    ) -> pl.LazyFrame:
        """Gets the last snapshot of the given table

        This method filters the data to include only the rows from the most recent snapshot time.

        Parameters
        ----------
        data : Optional[pl.LazyFrame], optional
            If provided, subsets to just that dataframe, by default None
        table : Literal['model_data', 'predictor_data', 'combined_data'], optional
            If provided, specifies the table to get data from, by default "model_data"

        Returns
        -------
        pl.LazyFrame
            A LazyFrame containing only the rows from the most recent snapshot time
        """
        if data is None and not hasattr(self.datamart, table):
            raise ValueError(f"{table} not available in the datamart")

        df: pl.LazyFrame = data if data is not None else getattr(self.datamart, table)
        if df.collect_schema()["SnapshotTime"] == pl.Null:
            return df

        return df.filter(
            # For safety consider to .over("ModelID"), if product improves so snapshots
            # get written not in bulk but per model? Downside is that
            # very old model IDs that never got used anymore would still show up.
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
        return (
            self.last(data=model_df)
            .join(self.last(data=predictor_df), on="ModelID", suffix="Bin")
            .rename({"PerformanceBin": "PredictorPerformance"}, strict=False)
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
                (pl.col("EntryType") == "Active")
                if active_only
                else (pl.col("EntryType") != "Classifier")
            ),
            query,
        )
        unique_predictors = df.select(pl.col("PredictorName").unique()).collect()[
            "PredictorName"
        ]
        if isinstance(by, str):
            by_col = pl.col(by)
            by_name = by
        else:
            by_col = by
            by_name = by.meta.output_name()
        action_predictor = by_col.meta.root_names() + ["PredictorName"]
        q = (
            (
                df.filter(pl.col("ResponseCount") > 0)
                .with_columns(
                    pl.col("PredictorPerformance").fill_nan(0.5),  # should we do this?
                )
                .unique(subset=action_predictor, keep="first")
                .group_by(action_predictor)
                .agg(
                    cdh_utils.weighted_average_polars(
                        "PredictorPerformance", "ResponseCountBin"
                    )
                )
            )
            .group_by(by_col)
            .agg(
                [
                    (
                        pl.when(pl.col("PredictorName") == predictor)
                        .then(pl.col("PredictorPerformance"))
                        .otherwise(pl.lit(0.5))
                        .alias(predictor)
                    )
                    for predictor in unique_predictors
                ]
            )
            .with_columns(pl.all().exclude(by_name).list.max())
        ).sort(pl.mean_horizontal(pl.all().exclude(by_name)), descending=True)

        column_order = (
            q.select(pl.all().exclude(by_name).mean())
            .collect()
            .transpose(include_header=True)
        ).sort("column_0", descending=True)["column"]

        if top_predictors:
            column_order = column_order.head(top_predictors)
        if top_groups:
            q = q.head(top_groups)

        return q.select(by_name, *column_order)

    # TODO: how is this used, where? Overlap with other summary function?
    # should it also have Performance?

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
            self.datamart.context_keys[: self.datamart.context_keys.index(by) + 1]
            if by != "ModelID"
            else by
        )

        return (
            df.group_by(group_by)
            .agg(
                pl.len().alias("count"),
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

    # TODO: how is this used? Shouldn't it be just a group-by on predictorname + category ? May need to be refactored a bit

    def predictor_counts(
        self,
        *,
        facet: str = "Configuration",
        by: str = "Type",
        query: Optional[QUERY] = None,
    ) -> pl.LazyFrame:
        """Returns the count of each predictor grouped by a certain column

        Parameters
        ----------
        facet : str, optional
            The column to use as a secondary grouping dimension, by default "Configuration"
        by : str, optional
            The column to group the data by, by default "Type"
        query : Optional[QUERY], optional
            A query to apply to the data, by default None

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with one row per predictor and 'by' combination, containing:
            - Name - The action name
            - EntryType - The entry type (Active, Inactive, etc.)
            - by - The column specified in the 'by' parameter
            - facet - The column specified in the 'facet' parameter
            - PredictorCount - The number of unique predictors for this combination
        """
        df = (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="combined_data"), query=query
            )
            .select("Name", "EntryType", "PredictorName", by, facet)
            .filter(pl.col("PredictorName") != "Classifier")
            .group_by(pl.all().exclude("PredictorName"))
            .agg(PredictorCount=pl.n_unique("PredictorName"))
        )

        overall = (
            df.group_by(pl.all().exclude(["PredictorName", by, "PredictorCount"]))
            .agg(pl.sum("PredictorCount"))
            .with_columns(pl.lit("Overall").alias(by))
        )

        # Collect schema once and use it for casting both DataFrames
        schema = df.collect_schema()

        return (
            pl.concat([df, overall.select(schema.names()).cast(schema)])
            .with_columns(
                pl.col("PredictorCount").cast(pl.Int64), cs.categorical().cast(pl.Utf8)
            )
            .sort(["Name", "EntryType", by, facet, "PredictorCount"])
        )

    @staticmethod
    def _top_n(
        df: pl.DataFrame,
        top_n: int,
        metric: str = "PredictorPerformance",
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

    def _adm_model_summary(
        self,
        *,
        query: Optional[QUERY] = None,
        by_period: Optional[str],
        by_channel: bool = False,
        debug: bool = False,
        custom_channels: Optional[Dict[str, str]] = None,
    ) -> pl.LazyFrame:
        custom_channels = custom_channels or {}

        def name_normalizer(x):
            return (
                pl.col(x)
                .cast(pl.Utf8)
                .str.replace_all(r"[ \-_]", "")
                .str.to_uppercase()
            )

        if self.datamart.model_data is None:
            raise ValueError("Model summaries needs model data")

        model_data = cdh_utils._apply_query(
            self.datamart.model_data, query=query, allow_empty=True
        )
        grouping = []

        if by_period:
            model_data = model_data.with_columns(
                pl.col("SnapshotTime").dt.truncate(by_period).alias("Period")
            )
            grouping += ["Period"]

        if by_channel:
            channelGroupMapping = (
                pl.concat(
                    [
                        pl.DataFrame(
                            {
                                "Channel": self.cdh_guidelines.standard_channels,
                                "ChannelGroup": self.cdh_guidelines.standard_channels,
                            }
                        ),
                        pl.DataFrame(
                            {
                                "Channel": list(custom_channels.keys()),
                                "ChannelGroup": list(custom_channels.values()),
                            }
                        ),
                    ],
                    how="diagonal",
                )
                .with_columns(normalizedChannel=name_normalizer("Channel"))
                .unique()
                .sort(["ChannelGroup", "Channel"])
            )

            model_data = (
                model_data.with_columns(
                    normalizedChannel=name_normalizer("Channel"),
                )
                .join(channelGroupMapping.lazy(), on="normalizedChannel", how="left")
                .with_columns(
                    ChannelDirectionGroup=pl.when(
                        pl.col("ChannelGroup").is_not_null()
                        & pl.col("Direction").is_not_null()
                        & pl.col("ChannelGroup").is_in(["Other", "Unknown", ""]).not_()
                    )
                    .then(pl.concat_str(["ChannelGroup", "Direction"], separator="/"))
                    .otherwise(pl.lit("Other")),
                )
            )
            grouping += ["Channel", "Direction", "ChannelDirectionGroup"]

        grouping = None if len(grouping) == 0 else grouping

        return (
            self._summarize_meta_info(grouping, model_data, debug=debug)
            .join(
                self._summarize_model_analytics(grouping, model_data, debug=debug),
                on=("literal" if grouping is None else grouping),
                nulls_equal=True,
                how="left",
            )
            .join(
                self._summarize_action_analytics(grouping, model_data, debug=debug),
                on=("literal" if grouping is None else grouping),
                nulls_equal=True,
                how="left",
            )
            .join(
                self._summarize_model_usage(
                    grouping,
                    model_data,
                    debug=debug,
                ),
                on=("literal" if grouping is None else grouping),
                nulls_equal=True,
                how="left",
            )
            .drop(["literal"] if grouping is None else [])
            .sort([] if grouping is None else grouping)
        )

    def _summarize_meta_info(
        self, grouping: Optional[List[str]], model_data: pl.LazyFrame, debug: bool
    ) -> pl.LazyFrame:
        return (
            model_data.group_by(grouping)
            .agg(
                pl.col("SnapshotTime").min().cast(pl.Date).alias("DateRange Min"),
                pl.col("SnapshotTime").max().cast(pl.Date).alias("DateRange Max"),
                pl.col("Configuration").cast(pl.Utf8).unique().sort(),
                (pl.col("SnapshotTime").max() - pl.col("SnapshotTime").min())
                .dt.total_seconds()
                .alias("Duration"),
            )
            .with_columns(
                pl.when(pl.col("Duration") == 0)
                .then(pl.duration(days=1).dt.total_seconds())
                .otherwise(pl.col("Duration"))
                .alias("Duration"),
                pl.col("Configuration").list.join(", "),
            )
        )

    def _summarize_model_analytics(
        self, grouping: Optional[List[str]], model_data: pl.LazyFrame, debug: bool
    ) -> pl.LazyFrame:
        return (
            model_data.group_by(([] if grouping is None else grouping) + ["ModelID"])
            .agg(
                (
                    pl.col("Positives").filter(Direction="Inbound").max()
                    - pl.col("Positives").filter(Direction="Inbound").min()
                ).alias("Positives Inbound"),
                (
                    pl.col("Positives").filter(Direction="Outbound").max()
                    - pl.col("Positives").filter(Direction="Outbound").min()
                ).alias("Positives Outbound"),
                (
                    pl.col("ResponseCount").filter(Direction="Inbound").max()
                    - pl.col("ResponseCount").filter(Direction="Inbound").min()
                ).alias("Responses Inbound"),
                (
                    pl.col("ResponseCount").filter(Direction="Outbound").max()
                    - pl.col("ResponseCount").filter(Direction="Outbound").min()
                ).alias("Responses Outbound"),
                pl.col("Positives").max().alias("TotalPositives"),
                pl.col("ResponseCount").max().alias("TotalResponseCount"),
                pl.col("Positives").max() - pl.col("Positives").min(),
                pl.col("ResponseCount").max() - pl.col("ResponseCount").min(),
                pl.col("Performance").mean(),  # ahum, not weighted?
            )
            .group_by(grouping)
            .agg(
                pl.sum(
                    "Positives",
                    "ResponseCount",
                    "TotalPositives",
                    "TotalResponseCount",
                    "Positives Inbound",
                    "Positives Outbound",
                    "Responses Inbound",
                    "Responses Outbound",
                ),
                (cdh_utils.weighted_performance_polars() * 100).alias("Performance"),
            )
            .with_columns(
                # applies to totals not delta
                isValid=(pl.col("TotalPositives") >= 200)
                & (pl.col("TotalResponseCount") >= 1000),
            )
            .drop([] if debug else ["ResponseCount", "Positives"])
        )

    def _summarize_action_analytics(
        self, grouping: Optional[List[str]], model_data: pl.LazyFrame, debug: bool
    ) -> pl.LazyFrame:
        if "Treatment" in self.datamart.context_keys:
            treatment_summary = (
                model_data.filter(pl.col("Treatment") != "")
                .filter(pl.col("Treatment").is_not_null())
                .group_by(grouping)
                .agg(
                    pl.len().alias("Treatments"),
                    pl.sum("IsUpdated").alias("Used Treatments"),
                )
            )

        action_summary = (
            (
                model_data.group_by(grouping).agg(
                    (
                        pl.col("Issue").n_unique()
                        if "Issue" in self.datamart.context_keys
                        else pl.lit(0)
                    ).alias("Issues"),
                    (
                        pl.concat_str(["Issue", "Group"], separator="/").n_unique()
                        if "Issue" in self.datamart.context_keys
                        and "Group" in self.datamart.context_keys
                        else pl.lit(0)
                    ).alias("Groups"),
                    pl.col("Name").n_unique().alias("Actions"),
                    pl.col("Name").filter("IsUpdated").n_unique().alias("Used Actions"),
                    pl.col("Name").unique().alias("AllActions"),
                    MinSnapshotTime=pl.col("SnapshotTime").min(),
                )
            )
            .collect()
            .lazy()
        )

        # Dropping the actions that are there from the very beginning would make interpretation rather difficult.
        # very_first_date = (
        #     self.datamart.first_action_dates.select(pl.col("FirstSnapshotTime").min())
        #     .collect()
        #     .item()
        # )

        new_action_summary = (
            (
                action_summary.select(
                    ([] if grouping is None else grouping)
                    + ["AllActions", "MinSnapshotTime"]
                )
                .explode("AllActions")
                .join_where(
                    self.datamart.first_action_dates,
                    pl.col("FirstSnapshotTime") >= pl.col("MinSnapshotTime"),
                    # may result in multiple rows...
                )
                .group_by(grouping)
                .agg(
                    pl.col("AllActions").unique(),
                    pl.col("Name")
                    .alias("NewActionsAtOrAfter")
                    # .filter(pl.col("FirstSnapshotTime") > very_first_date)
                    .list.explode()
                    .unique(),
                )
                .with_columns(
                    pl.col("AllActions")
                    .list.set_intersection(pl.col("NewActionsAtOrAfter"))
                    .alias("NewActionsList"),
                    pl.col("AllActions")
                    .list.set_intersection(pl.col("NewActionsAtOrAfter"))
                    .list.len()
                    .alias("New Actions"),
                )
            )
            .collect()
            .lazy()
        )

        action_summary = (
            action_summary.drop("MinSnapshotTime")
            .join(
                new_action_summary.drop("AllActions"),
                on=("literal" if grouping is None else grouping),
                how="left",
                nulls_equal=True,
            )
            .with_columns(pl.col("New Actions").fill_null(0))
            .drop([] if debug else ["NewActionsList", "NewActionsAtOrAfter"])
        )

        if "Treatment" in self.datamart.context_keys:
            return action_summary.join(
                treatment_summary,
                on=("literal" if grouping is None else grouping),
                nulls_equal=True,
                how="left",
            ).fill_null(0)
        else:
            return action_summary.with_columns(
                pl.lit(0).alias("Treatments"),
                pl.lit(0).alias("Used Treatments"),
            ).fill_null(0)

    def _summarize_model_usage(
        self,
        grouping: Optional[List[str]],
        model_data: pl.LazyFrame,
        debug: bool,
    ) -> pl.LazyFrame:
        result = model_data.group_by(grouping).agg(
            self.cdh_guidelines.is_standard_configuration()
            .any(ignore_nulls=False)
            .alias("usesNBAD"),
            (pl.col("ModelTechnique") == "GradientBoost")
            .any(ignore_nulls=False)
            .alias("usesAGB"),
            # For debugging:
            pl.col("ModelTechnique").unique().sort(),
            pl.col("Configuration").unique().sort().alias("Configurations"),
        )

        if debug:
            return result
        else:
            return result.drop("ModelTechnique", "Configurations")

    def summary_by_channel(
        self,
        *,
        query: Optional[QUERY] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        window: Optional[Union[int, datetime.timedelta]] = None,
        by_period: Optional[str] = None,
        custom_channels: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Summarize ADM models per channel

        Parameters
        ----------
        query : Optional[QUERY], optional
            A query to apply to the data, by default None, so no filtering applied
        start_date : datetime.datetime, optional
            Start date of the summary period. If None (default) uses the end date minus the window, or if both absent, the earliest date in the data
        end_date : datetime.datetime, optional
            End date of the summary period. If None (default) uses the start date plus the window, or if both absent, the latest date in the data
        window : int or datetime.timedelta, optional
            Number of days to use for the summary period or an explicit timedelta. If None (default) uses the whole period. Can't be given if start and end date are also given.
        by_period : str, optional
            Optional additional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. Defaults to None.
        custom_channels : Dict[str, str], optional
            Optional dictionary mapping custom channel names to standard channel groups. Defaults to None.
        debug : bool, optional
            If True, enables debug mode for additional logging or outputs. Defaults to False.

        Returns
        -------
        pl.LazyFrame
            Dataframe with summary per channel (and optionally a period) with the following fields:

            Channel Identification:
            - Channel - The channel name
            - Direction - The direction (e.g., Inbound, Outbound)
            - ChannelDirection - Combined Channel/Direction (e.g., "Web/Inbound")
            - ChannelDirectionGroup - Standardized channel group with direction (e.g., "Web/Inbound")

            Time and Configuration Fields:
            - DateRange Min - The minimum date in the summary time range
            - DateRange Max - The maximum date in the summary time range
            - Duration - The duration in seconds between the minimum and maximum snapshot times
            - Configuration - A comma-separated list of model configuration names

            Performance Metrics:
            - Positives - The sum of positive responses across all models in the channel
            - Responses - The sum of all responses across all models in the channel
            - Performance - The weighted average performance across all models in the channel (50-100)
            - CTR - Click-through rate (Positives / Responses) in the channel
            - isValid - Boolean indicating if the channel has sufficient data (at least 200 positives and 1000 responses)

            Action Statistics:
            - Actions - The total number of unique actions in the channel
            - Used Actions - The number of unique actions that have been used (have responses)
            - New Actions - The number of new actions introduced in the period
            - Issues - The number of unique issues
            - Groups - The number of unique issue/group combinations

            Treatment Statistics:
            - Treatments - The total number of unique treatments
            - Used Treatments - The number of unique treatments

            Omnichannel Metrics:
            - OmniChannel - The overlap of actions with other channels (measure of Omni Channel capability)

            Technology Usage Indicators:
            - usesNBAD - Boolean indicating whether any standard NBAD configurations are used
            - usesAGB - Boolean indicating whether any Adaptive Generic Boosting (AGB) models are used
        """

        start_date, end_date = cdh_utils._get_start_end_date_args(
            self.datamart.model_data, start_date, end_date, window
        )

        if query is None:
            query = pl.col("SnapshotTime").is_between(start_date, end_date)
        else:
            query = pl.col("SnapshotTime").is_between(start_date, end_date) & query

        summary_by_channel = (
            self._adm_model_summary(
                query=query,
                by_period=by_period,
                by_channel=True,
                debug=debug,
                custom_channels=custom_channels,
            )
            .with_columns(
                Positives=pl.col("Positives Inbound") + pl.col("Positives Outbound"),
                Responses=pl.col("Responses Inbound") + pl.col("Responses Outbound"),
                CTR=(pl.col("Positives Inbound") + pl.col("Positives Outbound"))
                / (pl.col("Responses Inbound") + pl.col("Responses Outbound")),
            )
            .drop(
                "Positives Inbound",
                "Positives Outbound",
                "Responses Inbound",
                "Responses Outbound",
            )
        )

        omni_channel_summary = (
            summary_by_channel.filter(pl.col("isValid"))
            .group_by(None if by_period is None else "Period")
            .agg(
                pl.col("Channel"),
                pl.col("Direction"),
                pl.col("AllActions")
                .map_batches(cdh_utils.overlap_lists_polars, return_dtype=pl.Float64)
                .alias("OmniChannel"),
            )
            # collect/lazy seems to resolve some polars issues
            .collect()
            .lazy()
            .drop(["literal"] if by_period is None else [])
            .explode(["Channel", "Direction", "OmniChannel"])
        )

        return (
            summary_by_channel.drop(["AllActions"])
            .join(
                omni_channel_summary,
                on=([] if by_period is None else ["Period"]) + ["Channel", "Direction"],
                nulls_equal=True,
                how="left",
            )
            .with_columns(
                cs.categorical().cast(pl.Utf8),
                pl.format("{}/{}", pl.col("Channel"), pl.col("Direction")).alias(
                    "ChannelDirection"
                ),
            )
            .drop(
                []
                if debug
                else (
                    ["TotalPositives", "TotalResponseCount"]
                    + ([] if by_period is None else ["Period"])
                )
            )
            .sort("Channel", "Direction", "DateRange Min")
            .with_columns(pl.col("OmniChannel").cast(pl.Float64))
        )

    def summary_by_configuration(self) -> pl.LazyFrame:
        """
        Generates a summary of the ADM model configurations.

        This method provides an overview of model configurations, including information about
        the number of models, actions, treatments, and performance metrics.

        Returns
        -------
        pl.LazyFrame
            A Polars LazyFrame containing the configuration summary with the following fields:

            Configuration Information:
            - Configuration - The name of the model configuration
            - Channel - The channel name (if available in context keys)
            - Direction - The direction (if available in context keys)

            Model Information:
            - ModelID - The number of unique model IDs for this configuration

            Action Statistics:
            - Actions - The number of unique actions in this configuration
            - Unique Treatments - The number of unique treatments (if available)
            - Used for (Issues) - A comma-separated list of issues this configuration is used for (if available)

            Performance Metrics:
            - ResponseCount - The total number of responses for this configuration
            - Positives - The total number of positive responses for this configuration
            - ModelsPerAction - The ratio of models to actions (models per action)
            - Performance - The weighted average model performance

            Technology Usage Indicators:
            - usesNBAD - Boolean indicating whether any standard NBAD configurations are used
            - usesAGB - Boolean indicating whether any Adaptive Generic Boosting (AGB) models are used

        """

        action_dim_agg = [pl.col("Name").n_unique().alias("Actions")]
        if "Treatment" in self.datamart.context_keys:
            action_dim_agg += [
                pl.col("Treatment").n_unique().alias("Unique Treatments")
            ]
        else:
            action_dim_agg += [pl.lit(0).alias("Unique Treatments")]

        if "Issue" in self.datamart.context_keys:
            action_dim_agg += [
                pl.col("Issue").cast(pl.String).unique().alias("Used for (Issues)")
            ]

        group_by_cols = ["Configuration"] + [
            c for c in ["Channel", "Direction"] if c in self.datamart.context_keys
        ]

        configuration_summary = (
            self.last(table="model_data")
            .group_by(group_by_cols)
            .agg(
                self.cdh_guidelines.is_standard_configuration()
                .any(ignore_nulls=False)
                .alias("usesNBAD"),
                (pl.col("ModelTechnique") == "GradientBoost")
                .any(ignore_nulls=False)
                .alias("usesAGB"),
                pl.col("ModelID").n_unique(),
                *action_dim_agg,
                pl.sum(["ResponseCount", "Positives"]),
                cdh_utils.weighted_average_polars("Performance", "ResponseCount")
                * 100.0,
            )
            .with_columns(
                (pl.col("ModelID") / pl.col("Actions"))
                .round(2)
                .alias("ModelsPerAction"),
            )
            .sort(group_by_cols)
        )
        if "Issue" in self.datamart.context_keys:
            configuration_summary = configuration_summary.with_columns(
                pl.col("Used for (Issues)").list.unique().list.sort().list.join(", ")
            )

        return configuration_summary

    def predictors_global_overview(
        self,
    ) -> pl.LazyFrame:
        """
        Generate a global overview of all predictors across all models.

        This method provides a summary of predictor performance and characteristics
        across all models, including the number of responses, positives, and performance metrics.

        Returns
        -------
        pl.LazyFrame
            A Polars LazyFrame containing the global predictor overview with the following fields:

            - PredictorName - The name of the predictor
            - Response Count Min/Max - The total number of responses for this predictor
            - Positives - The total number of positive responses for this predictor
            - Min, Mean, Median, Max - The min, mean, median and max performance of the predictor (AUC)
        """

        data = self.last(table="predictor_data")

        global_overview = (
            data.filter(pl.col("EntryType") != "Classifier")
            .filter(BinIndex=1)
            .group_by("PredictorName", "PredictorCategory")
            .agg(
                [
                    # weighted performance
                    pl.min("ResponseCount").alias("Response Count Min"),
                    pl.max("ResponseCount").alias("Response Count Max"),
                    pl.col("ModelID")
                    .filter(EntryType="Active")
                    .n_unique()
                    .alias("Active in Models"),
                    (pl.min("Performance") * 100).alias("Min"),
                    (pl.mean("Performance") * 100).alias("Mean"),
                    (pl.median("Performance") * 100).alias("Median"),
                    (pl.max("Performance") * 100).alias("Max"),
                ]
            )
            .sort("PredictorName")
        )
        return global_overview

    def predictors_overview(
        self,
        model_id: Optional[str] = None,
        additional_aggregations: Optional[list] = None,
    ) -> Optional[pl.LazyFrame]:
        """
        Generate a summary of the last snapshot of predictor data.

        This method provides an overview of predictor performance and characteristics
        from the most recent snapshot, either for all models or for a specific model.

        Parameters
        ----------
        model_id : Optional[str], optional
            If provided, filters the data to include only predictors for the specified model ID.
            If None (default), includes predictors for all models.
        additional_aggregations : Optional[list], optional
            Additional aggregation expressions to include in the result.
            These will be added to the default aggregations.

        Returns
        -------
        pl.LazyFrame or None
            A Polars LazyFrame containing the predictor summary with the following fields:

            Identification:
            - ModelID - The model ID (only if model_id parameter is None)
            - PredictorName - The name of the predictor

            Status and Type:
            - EntryType - The entry type (Active, Inactive, etc.)
            - isActive - Boolean indicating if the predictor is active
            - Type - The predictor type
            - GroupIndex - The group index of the predictor

            Performance Metrics:
            - Responses - The number of responses for this predictor
            - Positives - The number of positive responses for this predictor
            - Univariate Performance - The univariate performance of the predictor (AUC)

            Binning Information:
            - Bins - The number of bins for this predictor
            - Missing % - The percentage of responses in the MISSING bin
            - Residual % - The percentage of responses in the RESIDUAL bin

            Returns None if the required data is not available or an error is encountered.
        """
        try:
            data = self.last(table="predictor_data")

            if model_id is not None:
                data = data.filter(pl.col("ModelID") == model_id)
                group_cols = ["PredictorName", "PredictorCategory"]
            else:
                group_cols = ["ModelID", "PredictorName", "PredictorCategory"]

            default_aggs = [
                pl.last("ResponseCount").cast(pl.Int64).alias("Responses"),
                pl.last("Positives").cast(pl.Int64),
                pl.last("EntryType"),
                (pl.last("EntryType") == "Active").alias("isActive"),
                pl.last("GroupIndex").cast(pl.Int16),
                pl.last("Type"),
                pl.last("Performance").cast(pl.Float32).alias("Univariate Performance"),
                pl.max("BinIndex").cast(pl.Int16).alias("Bins"),
                (
                    pl.col("BinResponseCount")
                    .filter(pl.col("BinType") == "MISSING")
                    .sum()
                    * 100
                    / pl.sum("BinResponseCount")
                )
                .cast(pl.Float64)
                .alias("Missing %"),
                (
                    pl.col("BinResponseCount")
                    .filter(pl.col("BinType") == "RESIDUAL")
                    .sum()
                    * 100
                    / pl.sum("BinResponseCount")
                )
                .cast(pl.Float64)
                .alias("Residual %"),
            ]

            if additional_aggregations is not None:
                default_aggs.extend(additional_aggregations)

            result = data.group_by(group_cols).agg(*default_aggs)
            result = result.sort(
                ["GroupIndex", "isActive", "Univariate Performance"],
                descending=[False, True, True],
                nulls_last=True,
            )

            return result
        except ValueError:  # TODO: @yusufuyanik1 really swallowing? https://en.wikipedia.org/wiki/Error_hiding
            return None

    def overall_summary(
        self,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        window: Optional[Union[int, datetime.timedelta]] = None,
        by_period: Optional[str] = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Overall ADM models summary. Only valid data is included.

        Parameters
        ----------
        start_date : datetime.datetime, optional
            Start date of the summary period. If None (default) uses the end date minus the window, or if both absent, the earliest date in the data
        end_date : datetime.datetime, optional
            End date of the summary period. If None (default) uses the start date plus the window, or if both absent, the latest date in the data
        window : int or datetime.timedelta, optional
            Number of days to use for the summary period or an explicit timedelta. If None (default) uses the whole period. Can't be given if start and end date are also given.
        by_period : str, optional
            Optional additional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. Defaults to None.
        debug : bool, optional
            If True, enables debug mode for additional logging or outputs. Defaults to False.

        Returns
        -------
        pl.LazyFrame
            Summary across all valid ADM models as a dataframe with the following fields:

            Time and Configuration Fields:
            - DateRange Min - The minimum date in the snapshot time range
            - DateRange Max - The maximum date in the snapshot time range
            - Duration - The duration in seconds between the minimum and maximum snapshot times
            - Configuration - A comma-separated list of unique model configurations

            Performance Metrics:
            - Positives Inbound - The sum of positive responses across all models in the inbound channels
            - Positives Outbound - The sum of positive responses across all models in the outbound channels
            - Responses Inbound - The sum of all responses across all models in the inbound channels
            - Responses Outbound - The sum of all responses across all models in the outbound channels
            - Performance - The weighted average performance across all models (50-100)

            Action Statistics:
            - Actions - The total number of unique actions
            - Used Actions - The number of unique actions that have been used (have responses)
            - New Actions - The number of new actions introduced in the period
            - Issues - The number of unique issues
            - Groups - The number of unique issue/group combinations

            Treatment Statistics:
            - Treatments - The total number of unique treatments
            - Used Treatments - The number of unique treatments that have been used

            Channel Statistics:
            - Number of Valid Channels - The count of valid channels (channels with sufficient data)
            - Minimum Channel Performance - The performance of the channel with lowest performance
            - Channel with Minimum Performance - The channel/direction group with the lowest performance
            - OmniChannel - The average overlap of actions across channels (measure of Omni Channel capability)

            Technology Usage Indicators:
            - usesNBAD - Boolean indicating whether standard NBAD configurations are used
            - usesAGB - Boolean indicating whether any Adaptive Gradient Boosting (AGB) models are used

            Note: A channel is considered "valid" if it has at least 200 positives and 1000 responses
        """

        start_date, end_date = cdh_utils._get_start_end_date_args(
            self.datamart.model_data, start_date, end_date, window
        )

        overall_summary = (
            self._adm_model_summary(
                query=pl.col("SnapshotTime").is_between(start_date, end_date),
                by_period=by_period,
                by_channel=False,
                debug=debug,
            )
            .drop(
                "Configuration",
                "AllActions",
                "isValid",
            )
            .collect()
            .lazy()
        )

        best_worst_channel_summary = (
            self._adm_model_summary(
                query=pl.col("SnapshotTime").is_between(start_date, end_date),
                by_period=by_period,
                by_channel=True,
                debug=True,  # this gives us Period
            )
            .filter(pl.col("isValid"))
            .group_by(None if by_period is None else "Period")
            .agg(
                pl.len().alias("Number of Valid Channels"),
                pl.col("Performance").min().alias("Minimum Channel Performance"),
                pl.col("ChannelDirectionGroup")
                .top_k_by("Performance", 1, reverse=True)
                .first()
                .alias("Channel with Minimum Performance"),
                pl.col("AllActions")
                .map_batches(
                    cdh_utils.overlap_lists_polars,
                    returns_scalar=False,
                    return_dtype=pl.Float64,
                )
                .mean()
                .alias("OmniChannel"),
            )
            .drop(["literal"] if by_period is None else [])
            .collect()
            .lazy()
        )

        if by_period is None:
            return (
                pl.concat(
                    [overall_summary, best_worst_channel_summary],
                    how="horizontal",
                )
                .with_columns(
                    cs.categorical().cast(pl.Utf8),
                    pl.col("Number of Valid Channels").fill_null(0),
                )
                .drop([] if debug else ["TotalPositives", "TotalResponseCount"])
            )

        else:
            return (
                overall_summary.join(
                    best_worst_channel_summary,
                    on="Period",
                    nulls_equal=True,
                    how="left",
                )
                .with_columns(
                    cs.categorical().cast(pl.Utf8),
                    pl.col("Number of Valid Channels").fill_null(0),
                )
                .sort("DateRange Min")
                .drop(
                    [] if debug else ["TotalPositives", "TotalResponseCount", "Period"]
                )
            )

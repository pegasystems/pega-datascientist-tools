__all__ = ["Aggregates"]
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

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
                pl.col("PredictorName") != "Classifier"
            ),
            query,
        )
        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")
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
            .with_columns(pl.col("PredictorCount").cast(pl.Int64))
            .sort(["Name", "EntryType", by, facet])
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
        by_period: Optional[str],
        by_channel: bool,
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

        model_data = self.datamart.model_data
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
            self._summarize_meta_info(grouping, model_data)
            .join(
                self._summarize_model_analytics(grouping, model_data),
                on=("literal" if grouping is None else grouping),
                join_nulls=True,
                how="left",
            )
            .join(
                self._summarize_action_analytics(grouping, model_data),
                on=("literal" if grouping is None else grouping),
                join_nulls=True,
                how="left",
            )
            .join(
                self._summarize_model_usage(
                    grouping, model_data, self.cdh_guidelines.standard_configurations
                ),
                on=("literal" if grouping is None else grouping),
                join_nulls=True,
                how="left",
            )
            .drop(["literal"] if grouping is None else [])
            .sort([] if grouping is None else grouping)
        )

    def _summarize_meta_info(
        self, grouping: Optional[List[str]], model_data: pl.LazyFrame
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
        self, grouping: Optional[List[str]], model_data: pl.LazyFrame
    ) -> pl.LazyFrame:
        return (
            model_data.group_by(([] if grouping is None else grouping) + ["ModelID"])
            .agg(
                pl.col("Positives").max() - pl.col("Positives").min(),
                pl.col("ResponseCount").max() - pl.col("ResponseCount").min(),
                pl.col("Performance").mean(),
            )
            .group_by(grouping)
            .agg(
                pl.sum("Positives"),
                pl.sum("ResponseCount").alias("Responses"),
                (cdh_utils.weighted_performance_polars() * 100).alias("Performance"),
            )
            .with_columns(
                CTR=pl.col("Positives") / pl.col("Responses"),
                isValid=(pl.col("Positives") > 200) & (pl.col("Responses") > 1000),
            )
        )

    def _summarize_action_analytics(
        self, grouping: Optional[List[str]], model_data: pl.LazyFrame
    ) -> pl.LazyFrame:
        if "Treatment" in self.datamart.context_keys:
            treatment_summary = (
                model_data.group_by(
                    ([] if grouping is None else grouping) + ["Name", "Treatment"]
                )
                .agg(
                    (
                        pl.col("ResponseCount").max() > pl.col("ResponseCount").min()
                    ).alias("is_used"),
                )
                .filter(pl.col("Treatment") != "")
                .filter(pl.col("Treatment").is_not_null())
                .group_by(grouping)
                .agg(
                    pl.len().alias("Treatments"),
                    pl.sum("is_used").alias("Used Treatments"),
                )
            )

        action_summary = (
            model_data.group_by(
                ([] if grouping is None else grouping)
                + ["Name"]
                + (["Issue"] if "Issue" in self.datamart.context_keys else [])
                + (["Group"] if "Group" in self.datamart.context_keys else [])
            )
            .agg(
                (pl.col("ResponseCount").max() > pl.col("ResponseCount").min()).alias(
                    "is_used"
                ),
            )
            .group_by(grouping)
            .agg(
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
                pl.len().alias("Actions"),
                pl.sum("is_used").alias("Used Actions"),
                pl.col("Name").unique().alias("AllActions"),
            )
        )

        if "Treatment" in self.datamart.context_keys:
            return action_summary.join(
                treatment_summary,
                on=("literal" if grouping is None else grouping),
                join_nulls=True,
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
        standard_configurations: List[str],
    ) -> pl.LazyFrame:
        standard_configurations_set = set([x.upper() for x in standard_configurations])

        return model_data.group_by(grouping).agg(
            pl.col("Configuration")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .is_in(standard_configurations_set)
            .any(ignore_nulls=False)
            .alias("usesNBAD"),
            (pl.col("ModelTechnique") == "GradientBoost")
            .any(ignore_nulls=False)
            .alias("usesAGB"),
        )

    def summary_by_channel(
        self,
        by_period: Optional[str] = None,
        custom_channels: Optional[Dict[str, str]] = None,
    ) -> pl.LazyFrame:
        """Summarize ADM models per channel

        Parameters
        ----------
        by_period : str, optional
            Optional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. If provided, creates a new Period column with the truncated date/time. Defaults to None.

        Returns
        -------
        pl.LazyFrame
            Dataframe with summary per channel (and optionally a period)
        """

        summary_by_channel = self._adm_model_summary(
            by_period=by_period, by_channel=True, custom_channels=custom_channels
        )

        omni_channel_summary = (
            (
                summary_by_channel.filter(pl.col("isValid"))
                .group_by(None if by_period is None else "Period")
                .agg(
                    pl.col("Channel"),
                    pl.col("Direction"),
                    pl.col("AllActions")
                    .map_batches(cdh_utils.overlap_lists_polars)
                    .alias("OmniChannel"),
                )
            )
            .collect()
            .lazy()
        )  # collect/lazy just to help zoom in into issues earlier

        omni_channel_summary = omni_channel_summary.drop(
            ["literal"] if by_period is None else []
        ).explode(["Channel", "Direction", "OmniChannel"])

        return (
            summary_by_channel.drop(["AllActions"])
            .join(
                omni_channel_summary,
                on=([] if by_period is None else ["Period"]) + ["Channel", "Direction"],
                join_nulls=True,
                how="left",
            )
            .with_columns(
                cs.categorical().cast(pl.Utf8),
                pl.format("{}/{}", pl.col("Channel"), pl.col("Direction")).alias(
                    "ChannelDirection"
                ),
            )
            .sort(["Channel", "Direction"] + ([] if by_period is None else ["Period"]))
        )

    def summary_by_configuration(self) -> pl.DataFrame:
        """
        Generates a summary of the ADM model configurations.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame containing the configuration summary.
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
                [
                    pl.when((pl.col("ModelTechnique") == "GradientBoost").any())
                    .then(pl.lit("Yes"))
                    .when(pl.col("ModelTechnique").is_null().any())
                    .then(pl.lit("Unknown"))
                    .otherwise(pl.lit("No"))
                    .alias("AGB")
                ]
                + [
                    pl.col("ModelID").n_unique(),
                ]
                + action_dim_agg
                + [pl.sum(["ResponseCount", "Positives"])],
            )
            .with_columns(
                [
                    # pl.col("Configuration")
                    # .is_in(standardNBADNames.keys())
                    # .alias("Standard in NBAD Framework"),
                    (pl.col("ModelID") / pl.col("Actions"))
                    .round(2)
                    .alias("ModelsPerAction"),
                ]
            )
            .sort(group_by_cols)
        )
        if "Issue" in self.datamart.context_keys:
            configuration_summary = configuration_summary.with_columns(
                pl.col("Used for (Issues)").list.unique().list.sort().list.join(", ")
            )

        return configuration_summary

    def predictors_overview(self) -> Optional[pl.DataFrame]:
        """
        Generate a summary of the last snapshot of predictor data.

        This method creates a summary of predictor data by joining the last snapshots
        of predictor_data and model_data, then performing various aggregations and
        calculations. It excludes the "Classifier" predictor from the analysis.

        Returns
        -------
        pl.DataFrame or None
            A Polars DataFrame containing the predictor summary if successful,
            None if the required data is not available.
        """
        try:
            model_identifiers = ["Configuration"] + self.datamart.context_keys
            predictor_summary = (
                self.last(table="combined_data")
                .filter(pl.col("EntryType") != "Classifier")
                .group_by(model_identifiers + ["ModelID", "PredictorName"])
                .agg(
                    pl.first("Type"),
                    pl.first("Performance"),
                    pl.first("EntryType"),
                    pl.count("BinIndex").alias("Bin Count"),
                    pl.first("Positives"),
                    pl.col("BinResponseCount")
                    .filter(pl.col("BinType") == "MISSING")
                    .sum()
                    .alias("Missing Bin Responses"),
                    pl.first("ResponseCount"),
                )
                .fill_null(0)
                .fill_nan(0)
                .with_columns(
                    pl.col("Bin Count").cast(pl.Int16),
                    pl.col("Positives").cast(pl.Int64),
                    pl.col("ResponseCount").cast(pl.Int64),
                )
            )

            return predictor_summary
        except ValueError:  # really? swallowing?
            return None

    def overall_summary(self, by_period: str = None) -> pl.LazyFrame:
        """Overall ADM models summary. Only valid data is included.

        Parameters
        ----------
        custom_channels : Dict[str, str], optional
            Optional list with custom channel/direction name mappings. Defaults to None.
        by_period : str, optional
            Optional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. If provided, creates a new Period column with the truncated date/time. Defaults to None.

        Returns
        -------
        pl.LazyFrame
            Summary across all valid ADM models as a dataframe
        """
        overall_summary = self._adm_model_summary(
            by_period=by_period,
            by_channel=False,
        ).drop(["Configuration", "AllActions", "CTR", "isValid"])

        best_worst_channel_summary = (
            self._adm_model_summary(by_period=by_period, by_channel=True)
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
                .map_batches(cdh_utils.overlap_lists_polars, returns_scalar=False)
                .alias("OmniChannel"),
            )
            .with_columns(pl.col("OmniChannel").list.mean())
            .drop(["literal"] if by_period is None else [])
        )

        if by_period is None:
            return pl.concat(
                [overall_summary, best_worst_channel_summary],
                how="horizontal",
            ).with_columns(
                cs.categorical().cast(pl.Utf8),
            )
        else:
            return (
                overall_summary.join(
                    best_worst_channel_summary, on="Period", join_nulls=True, how="left"
                )
                .with_columns(
                    cs.categorical().cast(pl.Utf8),
                )
                .sort("Period")
            )

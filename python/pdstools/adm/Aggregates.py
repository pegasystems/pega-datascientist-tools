__all__ = ["Aggregates"]
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

        custom_channels = custom_channels or {}

        def name_normalizer(x):
            return (
                pl.col(x)
                .cast(pl.Utf8)
                .str.replace_all(r"[ \-_]", "")
                .str.to_uppercase()
            )

        directionMapping = pl.DataFrame(
            {
                "Direction": self.cdh_guidelines.standard_directions,
                "DirectionGroup": self.cdh_guidelines.standard_directions,
            }
        ).with_columns(normalizedDirection=name_normalizer("Direction"))

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

        context_keys = self.datamart.context_keys
        action_identifier = [
            item for item in ["Issue", "Group", "Name"] if item in context_keys
        ]
        actionIdentifierExpr = pl.concat_str(action_identifier, separator="/")

        has_treatment = "Treatment" in context_keys
        treatment_identifier = action_identifier + (
            ["Treatment"] if has_treatment else []
        )
        treatmentIdentifierExpr = (
            pl.concat_str(treatment_identifier, separator="/")
            if has_treatment
            else pl.lit("")
        )

        activeActionExpr = (pl.col("ResponseCount").sum() > 0).over(action_identifier)
        activeTreatmentExpr = (
            (
                (pl.col("ResponseCount").sum() > 0)
                & (pl.col("Treatment").is_not_null())
            ).over(treatment_identifier)
            if has_treatment
            else pl.lit(False)
        )

        period_expr = (
            [
                pl.col("SnapshotTime")
                .dt.truncate(by_period)
                .cast(pl.Date)
                .alias("Period")
            ]
            if by_period
            else []
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
            .join(channelGroupMapping.lazy(), on="normalizedChannel", how="left")
            .join(directionMapping.lazy(), on="normalizedDirection", how="left")
            .with_columns(
                ChannelDirectionGroup=pl.when(
                    pl.col("ChannelGroup").is_not_null()
                    & pl.col("DirectionGroup").is_not_null()
                    & pl.col("ChannelGroup").is_in(["Other", "Unknown", ""]).not_()
                    & pl.col("DirectionGroup").is_in(["Other", "Unknown", ""]).not_()
                )
                .then(pl.concat_str(["ChannelGroup", "DirectionGroup"], separator="/"))
                .otherwise(pl.lit("Other")),
            )
            .group_by(
                ["Channel", "Direction", "ChannelDirectionGroup"]
                + (["Period"] if by_period else [])
            )
            .agg(
                pl.col("SnapshotTime").min().cast(pl.Date).alias("DateRange Min"),
                pl.col("SnapshotTime").max().cast(pl.Date).alias("DateRange Max"),
                pl.sum(["Positives", "ResponseCount"]),
                (cdh_utils.weighted_performance_polars() * 100).alias("Performance"),
                pl.col("Configuration").cast(pl.Utf8),
                pl.col("Configuration")
                .cast(pl.Utf8)
                .str.to_uppercase()
                .is_in([x.upper() for x in self.cdh_guidelines.standard_configurations])
                .alias("isNBADModelConfiguration"),
                (pl.col("ModelTechnique") == "GradientBoost")
                .any(ignore_nulls=False)
                .alias("usesAGB"),
                (pl.col("ModelTechnique") == "GradientBoost")
                .all(ignore_nulls=False)
                .alias("usesAGBOnly"),
                actionIdentifierExpr.drop_nulls()
                .n_unique()
                .alias("Total Number of Actions"),
                (
                    treatmentIdentifierExpr.drop_nulls().n_unique()
                    if has_treatment
                    else pl.lit(0)
                ).alias("Total Number of Treatments"),
                # TODO use last update property instead
                actionIdentifierExpr.filter(pl.col("isUsedAction"))
                .drop_nulls()
                .n_unique()
                .alias("Used Actions"),
                (
                    treatmentIdentifierExpr.filter(pl.col("isUsedTreatment"))
                    .drop_nulls()
                    .n_unique()
                    if has_treatment
                    else pl.lit(0)
                ).alias("Used Treatments"),
                # keep lists of unique values for aggregation over channels
                AllIssues=(
                    pl.col("Issue").unique() if "Issue" in context_keys else pl.lit([])
                ),
                AllGroups=(
                    pl.concat_str(["Issue", "Group"], separator="/").unique()
                    if "Issue" in context_keys and "Group" in context_keys
                    else pl.lit([])
                ),
                AllActions=actionIdentifierExpr.unique(),
                AllTreatments=(
                    treatmentIdentifierExpr.unique() if has_treatment else pl.lit([])
                ),
                AllUsedActions=actionIdentifierExpr.filter(
                    pl.col("isUsedAction")
                ).unique(),
                AllUsedTreatments=(
                    treatmentIdentifierExpr.filter(pl.col("isUsedTreatment")).unique()
                    if has_treatment
                    else pl.lit([])
                ),
            )
            .with_columns(
                pl.col("Used Actions").fill_null(0),
                pl.col("Used Treatments").fill_null(0),
                ChannelDirection=pl.concat_str(["Channel", "Direction"], separator="/"),
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
                usesNBADOnly=pl.col("isNBADModelConfiguration").list.all(),
            )
            .sort(
                ["ChannelDirectionGroup", "Channel", "Direction"]
                + (["Period"] if by_period else [])
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
            CTR=pl.col("Positives") / pl.col("ResponseCount"),
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
                self.last(table="predictor_data")
                .filter(pl.col("PredictorName") != "Classifier") # TODO not name, there is a type
                .join(
                    self.last(table="model_data")
                    .select(["ModelID"] + model_identifiers)
                    .unique(),
                    on="ModelID",
                    how="left",
                )
                .group_by(model_identifiers + ["ModelID", "PredictorName"])
                .agg(
                    pl.first("Type"),
                    pl.first("Performance"),
                    pl.count("BinIndex").alias("Bins"),
                    pl.col("BinResponseCount")
                    .filter(pl.col("BinType") == "MISSING")
                    .sum()
                    .alias("Missing"),
                    pl.col("BinResponseCount")
                    .filter(pl.col("BinType") == "RESIDUAL")
                    .sum()
                    .alias("Residual"),
                    pl.first("Positives"),
                    pl.first("ResponseCount"),
                )
                .group_by(model_identifiers + ["PredictorName"])
                .agg(
                    pl.first("Type"),
                    cdh_utils.weighted_average_polars("Performance", "ResponseCount"),
                    cdh_utils.weighted_average_polars("Bins", "ResponseCount"),
                    ((pl.sum("Missing") / pl.sum("ResponseCount")) * 100).alias(
                        "Missing %"
                    ),
                    ((pl.sum("Residual") / pl.sum("ResponseCount")) * 100).alias(
                        "Residual %"
                    ),
                    pl.sum("Positives"),
                    pl.sum("ResponseCount").alias("Responses"),
                )
                .fill_null(0)
                .fill_nan(0)
                .with_columns(pl.col("Bins").cast(pl.Int16))
            )

            return predictor_summary
        except ValueError:  # really? swallowing?
            return None

    def overall_summary(
        self, custom_channels: Dict[str, str] = None, by_period: str = None
    ) -> pl.LazyFrame:
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
        totalTreatments = (
            pl.col("AllTreatments").list.explode().drop_nulls().n_unique()
            if "Treatment" in self.datamart.model_data.collect_schema().names()
            else pl.lit(0)
        )
        totalUsedTreatments = (
            pl.col("AllUsedTreatments").list.explode().drop_nulls().n_unique()
            if "Treatment" in self.datamart.model_data.collect_schema().names()
            else pl.lit(0)
        )

        # # Re-calculating here because the use of NBAD in the channel
        # # summary does not currently take into account the omni adaptive model
        # usesNBAD = (
        #     self.datamart.model_data.select(
        #         pl.col("Configuration")
        #         .cast(pl.Utf8)
        #         .str.to_uppercase()
        #         .is_in(self.cdh_guidelines.standard_configurations)
        #         .any()
        #     )
        #     .collect()
        #     .item()
        # )

        # usesNBADOnly = (
        #     self.datamart.model_data.select(
        #         pl.col("Configuration")
        #         .cast(pl.Utf8)
        #         .str.to_uppercase()
        #         .is_in(self.cdh_guidelines.standard_configurations)
        #         .all()
        #     )
        #     .collect()
        #     .item()
        # )

        return (
            self.summary_by_channel(
                custom_channels=custom_channels, by_period=by_period, keep_lists=True
            )
            .collect()
            # this is odd - maybe a Polars bug, when not doing this and the lazy later, getting a series length error on OmniChannel mean (not in prev versions)
            # I'm even thinking this is an issue with filter on lazy dfs, pl version is 1.10.0
            .filter(pl.col("isValid"))
            .group_by(["Period"] if by_period is not None else None)
            .agg(
                pl.col("DateRange Min").min(),
                pl.col("DateRange Max").max(),
                pl.len().alias("Number of Valid Channels"),
                cdh_utils.weighted_performance_polars().alias("Performance"),
                pl.col("Positives").sum(),
                pl.col("ResponseCount").sum(),
                pl.col("Performance")
                .filter((pl.col("Performance") == pl.col("Performance").min()))
                .first()
                .alias("Minimum Performance"),
                pl.col("ChannelDirection")
                .filter((pl.col("Performance") == pl.col("Performance").min()))
                .first()
                .alias("Channel with Minimum Performance"),
                pl.col("AllIssues")
                .list.explode()
                .drop_nulls()
                .n_unique()
                .alias("Issues"),
                pl.col("AllGroups")
                .list.explode()
                .drop_nulls()
                .n_unique()
                .alias("Groups"),
                pl.col("AllActions")
                .list.explode()
                .drop_nulls()
                .n_unique()
                .alias("Total Number of Actions"),
                totalTreatments.alias("Total Number of Treatments"),
                pl.col("AllUsedActions")
                .list.explode()
                .drop_nulls()
                .n_unique()
                .alias("Used Actions"),
                totalUsedTreatments.alias("Used Treatments"),
                # TODO there was something about OmniAdaptiveModel here - but I don't recall what was the issue
                pl.col("usesNBAD").any(),
                pl.col("usesNBADOnly").all(),
                pl.col("usesAGB").any(),
                pl.col("usesAGBOnly").all(),
                # pl.lit(usesNBAD).alias("usesNBAD"),
                # ((pl.len() > 0) & pl.lit(usesNBAD and usesNBADOnly)).alias(
                #     "usesNBADOnly"
                # ),
                pl.col("OmniChannel Actions").mean(),
            )
            .drop(["literal"] if by_period is None else [])  # created by null group
            .with_columns(CTR=(pl.col("Positives")) / (pl.col("ResponseCount")))
            .sort(["Period"] if by_period is not None else [])
        ).lazy()  # See above

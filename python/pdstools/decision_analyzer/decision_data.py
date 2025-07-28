from bisect import bisect_left
from functools import cached_property
from typing import List, Optional, Union, Dict
import warnings

import polars as pl
import polars.selectors as cs

from .data_read_utils import validate_columns
from .plots import Plot
from .utils import (
    NBADScope_Mapping,
    apply_filter,
    determine_extract_type,
    get_table_definition,
    gini_coefficient,
    rename_and_cast_types,
)


class DecisionAnalyzer:
    """
    Container data class for the raw decision data. Only one instance of this
    should exist and will be associated with the streamlit app state.

    It will keep a pointer to the raw interaction level data (as a
    lazy frame) but also has VBD-style aggregation(s) to speed things up.
    """

    # Raw data with only data-level pre-processing from the source. Should never
    # be accessed outside of this class.
    unfiltered_raw_decision_data: pl.LazyFrame = None

    # Interaction-level decision data with possibly the global filters applied. Try
    # to avoid accessing it directly, use class methods instead, but sometimes we
    # need to use this directly.
    decision_data: pl.LazyFrame = None

    # Decision data rolled up over interactions and customers, keeping a number of
    # meaningful aggregates useful for many analyses. Never access directly, always
    # use access method getPreaggregatedFilterView as this field is populated
    # on first use only, and reset when the global filters change.
    preaggregated_decision_data_filterview: pl.LazyFrame = None

    # Similar, but providing a view per stage, so strategy results can be duplicated.
    # Also never access directly, use getPreaggregatedRemainingView.
    preaggregated_decision_data_remainingview: pl.LazyFrame = None

    # Superset of all fields available for data filtering in various places.
    fields_for_data_filtering = [
        "pxDecisionTime",
        "pyConfigurationName",
        "pyChannel",
        "pyDirection",
        "pyIssue",
        "pyGroup",
        "pyName",
        "pyTreatment",  # or Treatment?
        "Stage",
        "ModelPositives",
        "ModelEvidence",
    ]

    def __init__(
        self,
        raw_data: pl.LazyFrame,
        level="StageGroup",
        sample_size=50000,
        mandatory_expr: Optional[pl.Expr] = None,
        additional_columns: Optional[Dict[str, pl.DataType]] = None,
    ):
        """Initialize DecisionAnalyzer with raw decision data.

        This class processes raw decision data (Explainability Extract) to create a comprehensive analysis framework
        for NBA. After light preprocessing it allows users to conduct various analysis on the data.

        For more information about Explainability Extract, see:
        https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/explainability-extract.html

        Parameters
        ----------
        raw_data : pl.LazyFrame
            Raw decision data containing interaction-level records from Explainability Extract.
        level : str, default "StageGroup"
            Granularity level for stage analysis. Options:
            - "StageGroup": Groups stages into categories (recommended)
            - "Stage": Individual stage-level analysis
        sample_size : int, default 50000
            Maximum number of unique interactions to sample for analysis. Larger values
            provide more statistical accuracy but slower performance. Minimum 1000.
        mandatory_expr : pl.Expr, optional
            Polars expression to create the `is_mandatory` column used in ranking.
            The expression should return True/False values that get converted to 1/0.
            Actions with is_mandatory=1 get FIRST rank in the ranking function.

            Example: `pl.col("pyIssue") == "Service"` results in:
            - Service actions: is_mandatory = 1 (ranked first)
            - Non-Service actions: is_mandatory = 0 (ranked by other criteria)

            Other examples:
            - `(pl.col("pyGroup") == "Credit") & (pl.col("Priority") > 0.8)`
            - `pl.col("pyName").is_in(["CriticalAction1", "CriticalAction2"])`
        additional_columns : Dict[str, pl.DataType], optional
            Additional columns to include in processing beyond the standard table definition.
            Dictionary mapping column names to their polars data types.

            Example: additional_columns = {"non_standard_column" : pl.Utf8}

        Notes
        -----
        The ranking function orders actions by: is_mandatory (desc) → Priority (desc) →
        StageOrder (desc) → pyIssue → pyGroup → pyName.

        If mandatory_expr is None, all actions get is_mandatory=0.
        The expression gets applied as: `raw_data.with_columns(is_mandatory=mandatory_expr)`

        Examples
        --------
        Basic usage:
        >>> analyzer = DecisionAnalyzer(raw_data)

        Make retention actions mandatory:
        >>> mandatory = pl.col("pyIssue") == "Retention"
        >>> decision_analyzer = DecisionAnalyzer(raw_data, mandatory_expr=mandatory)
        """
        self.plot = Plot(self)
        self.level = level  # Stage or StageGroup
        self.sample_size = sample_size
        # pxEngagement Stage present?
        self.extract_type = determine_extract_type(raw_data)
        # Get table definition and add any additional columns to it
        table_def = get_table_definition(self.extract_type)
        if additional_columns:
            for col_name, col_type in additional_columns.items():
                table_def[col_name] = {
                    "label": col_name,
                    "default": True,
                    "type": col_type,
                    "required": False,
                }
        # all columns are present?
        validation_result, validation_error = validate_columns(raw_data, table_def)
        self.validation_error = validation_error if not validation_result else None
        if not validation_result:
            warnings.warn(validation_error, UserWarning)
        # cast datatypes
        raw_data = rename_and_cast_types(df=raw_data, table_definition=table_def).sort(
            "pxInteractionID"
        )
        if mandatory_expr is not None:
            raw_data = raw_data.with_columns(is_mandatory=mandatory_expr)
        else:
            raw_data = raw_data.with_columns(is_mandatory=pl.lit(0))
        self.unfiltered_raw_decision_data = self.cleanup_raw_data(raw_data)
        self.resetGlobalDataFilters()
        # TODO subset against available fields in the data
        # TODO maybe we'll also need some aggregates per customer ID. Not certain, lets postpone, current dataset is not very representative.
        available_columns = set(
            self.unfiltered_raw_decision_data.collect_schema().names()
        )
        self.preaggregation_columns = {
            "pyIssue",
            "pyGroup",
            "pyName",
            "pyTreatment",
            "pyChannel",
            "pyDirection",
            "pxComponentName",
            # "ModelControlGroup",
            "day",
        }
        self.preaggregation_columns = self.preaggregation_columns.intersection(
            available_columns
        )

        self.max_win_rank = 5
        self.AvailableNBADStages = ["Arbitration", "Output"]

        if self.extract_type == "explainability_extract":
            columns_to_remove = {"pxComponentName"}
            self.preaggregation_columns -= columns_to_remove

            self.NBADStages_FilterView = self.AvailableNBADStages
            self.NBADStages_RemainingView = self.AvailableNBADStages

        # TODO: support human-friendly names, to show "Final" as "Presented" for example
        elif self.extract_type == "decision_analyzer":
            stage_df = (
                self.unfiltered_raw_decision_data.group_by(self.level)
                .agg(pl.min("StageOrder"))
                .collect()
            )
            if "Arbitration" not in stage_df[self.level] and self.level == "StageGroup":
                arb = pl.DataFrame(
                    {self.level: "Arbitration", "StageOrder": 3800},
                    schema=stage_df.schema,
                )
                stage_df = pl.concat([stage_df, arb])
            stage_df = stage_df.sort("StageOrder")
            self.AvailableNBADStages = stage_df.get_column(self.level).to_list()

    @cached_property
    def stages_from_arbitration_down(self):
        """
        All stages in the filter view starting at Arbitration. This initially
        will just be [Arbitration, Final] but as we get more stages in there
        may be more here.
        """
        return self.AvailableNBADStages[self.AvailableNBADStages.index("Arbitration") :]

    @cached_property
    def arbitration_stage(self):
        return self.sample.filter(
            pl.col(self.level).is_in(self.stages_from_arbitration_down)
        )

    @property
    def num_sample_interactions(self):
        """
        Number of unique interactions in the sample.
        Automatically triggers sampling if not yet calculated.
        """
        if not hasattr(self, "_num_sample_interactions"):
            # Trigger sample calculation to set _num_sample_interactions
            _ = self.sample
        return self._num_sample_interactions

    def _invalidate_cached_properties(self):
        """Resets the properties of the class. Needed for global filters."""
        cls = type(self)
        cached = {
            attr
            for attr in list(self.__dict__.keys())
            if (descriptor := getattr(cls, attr, None))
            if isinstance(descriptor, cached_property)
        }
        for attr in cached:
            del self.__dict__[attr]
        # Also reset num_sample_interactions so it gets recalculated with new filters
        if hasattr(self, "_num_sample_interactions"):
            delattr(self, "_num_sample_interactions")

    def applyGlobalDataFilters(
        self, filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None
    ):
        """
        Apply a global set of filters
        """
        self._invalidate_cached_properties()
        if filters is not None:
            self.decision_data = apply_filter(
                self.unfiltered_raw_decision_data, filters
            )

    def resetGlobalDataFilters(self):
        self.decision_data = self.unfiltered_raw_decision_data
        self._invalidate_cached_properties()

    @cached_property
    def getPreaggregatedFilterView(self):
        """Pre-aggregates the full dataset over customers and interactions providing
        a view of what is filtered at a stage.

        This pre-aggregation is pretty similar to what "VBD" does to interaction
        history. It aggregates over individual customers and interactions giving
        summary statistics that are sufficient to drive most of the analyses
        (but not all). The results of this pre-aggregation are much smaller
        than the original data and is expected to easily fit in memory. We therefore
        use polars caching to efficiently cache this.

        This "filter" view keeps the same organization as the decision analyzer data
        in that it records the actions that get filtered out at stages. From this
        a "remaining" view is easily derived.
        """
        num_samples = 1  # TODO: when > 1 this breaks the .explode() I'm doing in some places(thresholding analysis e.g.), need to solve
        stats_cols = ["pxDecisionTime", "Value", "Propensity", "Priority"]
        exprs = [
            pl.col("pxInteractionID")
            .where(pl.col("pxRank") <= i)
            .count()
            .alias(f"Win_at_rank{i}")
            for i in range(1, self.max_win_rank + 1)
        ] + [
            pl.min(stats_cols).name.suffix("_min"),
            pl.max(stats_cols).name.suffix("_max"),
            pl.col("Propensity", "Priority").sample(
                n=num_samples, with_replacement=True, shuffle=True
            ),
            pl.count().alias("Decisions"),
        ]

        self.preaggregated_decision_data_filterview = (
            self.decision_data.group_by(
                self.preaggregation_columns.union(
                    {self.level, "StageOrder", "pxRecordType"}
                )
            )
            .agg(exprs)
            .collect()
            .lazy()
        )
        return self.preaggregated_decision_data_filterview

    @cached_property
    def getPreaggregatedRemainingView(self):
        """Pre-aggregates the full dataset over customers and interactions providing a view of remaining offers.

        This pre-aggregation builds on the filter view and aggregates over
        the stages remaining.
        """
        self.preaggregated_decision_data_remainingview = (
            self.aggregate_remaining_per_stage(
                self.getPreaggregatedFilterView,
                self.preaggregation_columns,
                [
                    pl.sum("Decisions"),
                    pl.min("pxDecisionTime_min"),
                    pl.max("pxDecisionTime_max"),
                    pl.min("Value_min"),
                    pl.max("Value_max"),
                    # pl.col("Propensity").sample(
                    #     n=num_samples, with_replacement=True, shuffle=True
                    # ),  # a list sample values - for distribution plots
                    pl.first("Propensity"),
                    pl.min("Propensity_min"),
                    pl.max("Propensity_max"),
                    # pl.col("Priority")
                    # .sample(n=num_samples, with_replacement=True, shuffle=True)
                    # .alias("Priority"),
                    pl.first("Priority"),
                    pl.min("Priority_min"),
                    pl.max("Priority_max"),
                ]
                + [pl.sum(f"Win_at_rank{i}") for i in range(1, self.max_win_rank + 1)],
            )
            .collect()
            .lazy()
        )
        return self.preaggregated_decision_data_remainingview

    @cached_property
    def sample(self):
        """
        Create a sample of the data by taking the first 50,000 interactions.
        If there are fewer than 50,000 total interactions, no sampling is performed.
        """
        needed_columns = [
            "pxInteractionID",
            "pyChannel",
            "pyDirection",
            "pyIssue",
            "pyGroup",
            self.level,
            "StageOrder",
            "pxRank",
            "Priority",
            "Propensity",
            "Value",
            "Context Weight",
            "Levers",
            "pySubjectID",
            "pxRecordType",
            "pyName",
            "day",
            "is_mandatory",
        ]

        # Filter to only keep columns that exist in the data
        available_cols = set(self.decision_data.collect_schema().names())
        columns_to_keep = [col for col in needed_columns if col in available_cols]

        print("In sample function")
        total_interaction_count = (
            self.decision_data.set_sorted("pxInteractionID")
            .select(pl.n_unique("pxInteractionID").alias("unique_count"))
            .collect()
            .item()
        )
        print(f"Total interaction count: {total_interaction_count}")
        # Set num_sample_interactions attribute - use sample_size if we have more interactions than sample_size
        self.sample_size = min(total_interaction_count, self.sample_size)
        target_sample_size = self.sample_size
        sample_rate = min(1.0, target_sample_size / max(1, total_interaction_count))

        # Use hash-based sampling for efficiency - this is deterministic per interaction ID
        # but doesn't require collecting all unique IDs first, 15x faster than collecting 50k interactions first.
        df = (
            self.decision_data.select(columns_to_keep)
            .with_columns(
                [
                    (
                        pl.col("pxInteractionID").hash() % 1000 < 1000 * sample_rate
                    ).alias("_sample")
                ]
            )
            .filter(pl.col("_sample"))
            .drop("_sample")
            .collect()
            .shrink_to_fit()
            .lazy()
        )

        return df

    def getAvailableFieldsForFiltering(self, categoricalOnly=False):
        available_fields = []

        if categoricalOnly:
            schema_keys = (
                self.decision_data.select(cs.string(include_categorical=True))
                .collect_schema()
                .keys()
            )

            for field in self.fields_for_data_filtering:
                if field in schema_keys:
                    available_fields.append(field)
        else:
            all_columns = self.decision_data.columns

            for field in self.fields_for_data_filtering:
                if field in all_columns:
                    available_fields.append(field)
        return available_fields

    def cleanup_raw_data(self, df: pl.LazyFrame):
        """This method cleans up the raw data we read from parquet/S3/whatever.

        This likely needs to change as and when we get closer to product, to
        match what comes out of Pega. It does some modest type casting and
        potentially changing back some of the temporary column names that have
        been added to generate more data.
        """

        if "day" not in df.collect_schema().names():
            df = df.with_columns(day=pl.col("pxDecisionTime").dt.date())

        # Build ranking columns - include StageOrder only if it exists
        ranking_cols = ["is_mandatory", "Priority"]
        if "StageOrder" in df.collect_schema().names():
            ranking_cols.append("StageOrder")
        ranking_cols.extend(
            [
                pl.col("pyIssue").rank() * -1,
                pl.col("pyGroup").rank() * -1,
                pl.col("pyName").rank() * -1,
            ]
        )

        df = df.with_columns(
            pxRank=pl.struct(ranking_cols)
            .rank(descending=True, method="ordinal", seed=1)
            .over("pxInteractionID")
        )

        if self.extract_type == "explainability_extract":
            df = df.with_columns(
                pl.when(pl.col("pxRank") == 1)
                .then(pl.lit("Output"))
                .otherwise(pl.lit("Arbitration"))
                .alias(self.level),
                pl.when(pl.col("pxRank") == 1)
                .then(pl.lit(3800))
                .otherwise(pl.lit(10000))
                .alias("StageOrder"),
                pl.when(pl.col("pxRank") == 1)
                .then(pl.lit("OUTPUT"))
                .otherwise(pl.lit("FILTERED_OUT"))
                .alias("pxRecordType"),
            )

        preproc_df = (
            df.with_columns(pl.col(pl.Categorical).cast(pl.Utf8))
            .with_columns(
                pl.col(self.level).cast(pl.Categorical(ordering="physical")),
            )
            .filter(
                pl.col("pyName").is_not_null()
            )  # Why do we have null pyName values? this takes too much processing time
        )
        return preproc_df

    def getPossibleScopeValues(self):
        options = [
            col
            for col in NBADScope_Mapping.keys()
            if col in self.decision_data.collect_schema().names()
        ]
        return options

    def getPossibleStageValues(self):
        options = self.AvailableNBADStages
        # TODO figure out how to get the actual possible values, should be available from the enum directly
        # [
        #     stage
        #     for stage in self.NBADStages_FilterView
        #     if stage in df['pxEgagementStage'].categories
        # ]
        return options

    def getDistributionData(
        self,
        stage: str,
        grouping_levels: List[str],
        trend=False,
        additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
    ) -> pl.LazyFrame:
        distribution_data = (
            apply_filter(self.getPreaggregatedRemainingView, additional_filters)
            .filter(pl.col(self.level) == stage)
            .group_by(["day"] + [grouping_levels] if trend else grouping_levels)
            .agg(pl.sum("Decisions"))
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
        )

        return distribution_data

    # import streamlit as st
    # @st.cache_data
    def getFunnelData(
        self, scope, additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None
    ) -> pl.LazyFrame:
        # Apply filtering once to the pre-aggregated view
        filtered_df = apply_filter(self.getPreaggregatedFilterView, additional_filters)

        interaction_count_expr = (
            apply_filter(self.decision_data, additional_filters)
            .select("pxInteractionID")
            .unique()
            .count()
        )

        # Compute remaining actions funnel
        funnelData = self.aggregate_remaining_per_stage(
            df=filtered_df,
            group_by_columns=[scope],
            aggregations=pl.sum("Decisions").alias("count"),
        ).filter(pl.col("count") > 0)

        # Compute filtered funnel view
        filtered_funnel = (
            filtered_df.filter(pl.col("pxRecordType") == "FILTERED_OUT")
            .group_by([self.level, scope])
            .agg(count=pl.sum("Decisions"))
            .collect()
        )

        interaction_count = interaction_count_expr.collect().item()
        average_actions_expr = (
            pl.lit(interaction_count).alias("interaction_count"),
            (pl.col("count") / pl.lit(interaction_count)).alias("average_actions"),
        )

        return funnelData.with_columns(
            average_actions_expr
        ), filtered_funnel.with_columns(average_actions_expr)

    def getFilterComponentData(
        self, top_n, additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None
    ) -> pl.DataFrame:
        stages_actions_df = (
            apply_filter(self.getPreaggregatedFilterView, additional_filters)
            .filter(pl.col(self.level) != "Output")
            .group_by(self.level, "pxComponentName")
            .agg(pl.sum("Decisions").alias("Filtered Decisions"))
            .collect()
        )
        result = pl.concat(
            pl.collect_all(
                [
                    x.lazy()
                    .top_k(top_n, by="Filtered Decisions")
                    .sort("Filtered Decisions", descending=False)
                    for x in stages_actions_df.partition_by(self.level)
                ]
            )
        ).with_columns(pl.col(pl.Categorical).cast(pl.Utf8))

        return result

    def reRank(
        self,
        additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        overrides: List[pl.Expr] = [],
    ) -> pl.LazyFrame:
        """
        Calculates prio and rank for all PVCL combinations
        """
        # TODO: make generic to support situations where P, V, C or L are missing?
        # NOTE: Should we calculate for different stages?
        rank_exprs = [
            pl.struct(
                [
                    "is_mandatory",
                    x,
                    "StageOrder",
                    pl.col("pyIssue").rank() * -1,
                    pl.col("pyGroup").rank() * -1,
                    pl.col("pyName").rank() * -1,
                ]
            )
            .rank(descending=True, method="ordinal", seed=1)
            .over(["pxInteractionID"])
            .cast(pl.Int16)
            .alias(f"rank_{x.split('_')[1]}")
            for x in ["prio_PVCL", "prio_VCL", "prio_PCL", "prio_PVL", "prio_PVC"]
        ]

        rank_df = (
            apply_filter(self.sample, additional_filters)
            .with_columns(overrides)
            .filter(pl.col("Priority").is_not_null())
            .with_columns(
                prio_PVCL=(
                    pl.col("Propensity")
                    * pl.col("Value")
                    * pl.col("Context Weight")
                    * pl.col("Levers")
                ),
                prio_VCL=(
                    pl.col("Value") * pl.col("Context Weight") * pl.col("Levers")
                ),
                prio_PCL=(
                    pl.col("Propensity") * pl.col("Context Weight") * pl.col("Levers")
                ),
                prio_PVL=(pl.col("Propensity") * pl.col("Value") * pl.col("Levers")),
                prio_PVC=(
                    pl.col("Propensity") * pl.col("Value") * pl.col("Context Weight")
                ),
            )
            .with_columns(*rank_exprs)
        )

        return rank_df

    # TODO consider making his more generic, dropping the win_rank argument and
    # creating a larger result set for all possible ranks, with filtering only
    # in the UI.
    def get_win_loss_distribution_data(self, level, win_rank):
        win_col = f"Win_at_rank{win_rank}"
        group_level_win_losses = (
            self.getPreaggregatedRemainingView.filter(
                pl.col(self.level) == "Arbitration"
            )
            .group_by(level)
            .agg(Wins=pl.sum(win_col), Decisions=pl.sum("Decisions"))
            .with_columns(Losses=pl.col("Decisions") - pl.col("Wins"))
            .with_columns(
                Wins=pl.col("Wins") / pl.sum("Wins"),
                Losses=pl.col("Losses") / pl.sum("Losses"),
            )
        )

        group_level_win_losses = group_level_win_losses.melt(
            id_vars=level,
            value_vars=["Wins", "Losses"],
            variable_name="Status",
            value_name="Percentage",
        ).sort(["Status", "Percentage"], descending=True)

        return group_level_win_losses

    def get_optionality_data(self, df):
        """
        Finding the average number of actions per stage without trend analysis.
        We have to go back to the interaction level data, no way to
        use pre-aggregations unfortunately.
        """
        total_interactions = df.select("pxInteractionID").collect().unique().height
        expr = [
            pl.len().alias("nOffers"),
            pl.col("Propensity")
            .filter(pl.col("Propensity") < 0.5)
            .max()
            .alias("bestPropensity"),
        ]
        per_offer_count_and_stage = (
            self.aggregate_remaining_per_stage(
                df=df,
                group_by_columns=["pxInteractionID"],
                aggregations=expr,
            )
            .group_by(["nOffers", self.level])
            .agg(Interactions=pl.len(), AverageBestPropensity=pl.mean("bestPropensity"))
        )
        schema = per_offer_count_and_stage.collect_schema()
        zero_actions = (
            per_offer_count_and_stage.group_by("StageGroup")
            .agg(interaction_count=pl.sum("Interactions"))
            .with_columns(
                Interactions=(total_interactions - pl.col("interaction_count")).cast(
                    schema["Interactions"]
                ),
                AverageBestPropensity=pl.lit(0.0).cast(schema["AverageBestPropensity"]),
                nOffers=pl.lit(0).cast(schema["nOffers"]),
            )
            .drop("interaction_count")
        )
        optionality_data = pl.concat(
            [
                per_offer_count_and_stage,
                zero_actions.select(per_offer_count_and_stage.collect_schema().names()),
            ]
        ).sort("nOffers", descending=True)

        return optionality_data

    # @cached_property
    def get_optionality_data_with_trend(self, df=None):
        """
        Finding the average number of actions per stage with trend analysis.
        We have to go back to the interaction level data, no way to
        use pre-aggregations unfortunately.
        """
        if df is None:
            df = self.sample
        expr = [
            pl.len().alias("nOffers"),
            pl.col("Propensity")
            .filter(pl.col("Propensity") < 0.5)
            .max()
            .alias("bestPropensity"),
        ]
        optionality_data = (
            self.aggregate_remaining_per_stage(
                df=df,
                group_by_columns=["pxInteractionID", "day"],
                aggregations=expr,
            )
            .group_by(["nOffers", self.level, "day"])
            .agg(Interactions=pl.len(), AverageBestPropensity=pl.mean("bestPropensity"))
            .sort("nOffers", descending=True)
        )
        return optionality_data

    # @cached_property
    def get_optionality_funnel(self, df=None):
        if df is None:
            df = self.sample
        optionality_funnel = (
            self.get_optionality_data(df)
            .with_columns(
                pl.when(pl.col("nOffers") >= 7)
                .then(pl.lit("7+"))
                .otherwise(pl.col("nOffers").cast(str))
                .alias("available_actions")
            )
            .with_columns(
                pl.col("available_actions").cast(
                    pl.Enum(["0", "1", "2", "3", "4", "5", "6", "7+"])
                ),
            )
            .group_by(["StageGroup", "available_actions"])
            .agg(pl.sum("Interactions"))
            .sort(["StageGroup", "available_actions"])
        )
        return optionality_funnel

    def getActionVariationData(self, stage):
        data = pl.concat(
            [
                pl.DataFrame(
                    {
                        "ActionIndex": 0,
                        "pyName": "",
                        "Decisions": 0,
                        # self.level: stage,
                        "cumDecisions": 0,
                        "DecisionsFraction": 0.0,
                        "ActionsFraction": 0.0,
                    }
                ).with_columns(
                    pl.col("ActionIndex").cast(pl.UInt32),
                    pl.col("Decisions").cast(pl.UInt32),
                    pl.col("cumDecisions").cast(pl.UInt32),
                ),
                self.getDistributionData(
                    # TODO generalize so it does all context keys not just pyName
                    # use pl.struct like in first_level_stats function
                    stage=stage,
                    grouping_levels="pyName",
                    trend=False,
                )
                .with_columns(cumDecisions=pl.col("Decisions").cum_sum().cast(pl.Int32))
                .with_columns(
                    DecisionsFraction=(pl.col("cumDecisions") / pl.sum("Decisions")),
                    Decisions=pl.col("Decisions").cast(pl.Int32),
                )
                .collect()
                .with_row_count(  # TODO: renamed to with_row_index in newer polars version
                    "ActionIndex", 1
                )
                .with_columns(
                    ActionsFraction=pl.col("ActionIndex") / pl.count(),
                    ActionIndex=pl.col("ActionIndex").cast(pl.UInt32),
                    Decisions=pl.col("Decisions").cast(pl.UInt32),
                    cumDecisions=pl.col("cumDecisions").cast(pl.UInt32),
                ),
            ]
        )
        return data.lazy()

    # TODO: figure out how to main standard stage order, for now simply solved by sorting on counts
    def getABTestResults(self):
        tbl = (
            self.getPreaggregatedRemainingView.group_by(
                # TODO: we should include all the IA properties but they're not populated currently
                [self.level, "ModelControlGroup"]
            )
            .agg(pl.count())
            .collect()
            .pivot(
                index=self.level,
                values="count",
                columns=["ModelControlGroup"],
                sort_columns=True,
            )
            .with_columns(
                (pl.col("Control") / (pl.col("Test") + pl.col("Control"))).alias(
                    "Control Percentage"
                )
            )
            .sort("Test", descending=True)
        )
        return tbl

    # TODO, how to use cached functions with parameters, how can we cache this
    def getThresholdingData(
        self, fld, quantile_range=range(10, 100, 10)
    ):  # this is a very complicated function.
        thresholds_wide = (
            # Note: runs on pre-aggregated data - maybe should be using _min/_max for filtering instead
            self.getPreaggregatedFilterView.filter(
                pl.col(self.level).is_in(self.stages_from_arbitration_down)
            )
            .select(
                # TODO can probably code this up more efficiently
                [
                    pl.col(fld).explode().quantile(q / 100.0).alias(f"p{q}")
                    for q in quantile_range
                ]
                + [
                    (
                        (pl.col(fld).explode())
                        < (pl.col(fld).explode().quantile(q / 100.0))
                    )
                    .sum()
                    .alias(f"n{q}")
                    for q in quantile_range
                ]
                + [pl.lit("Arbitration").alias(self.level)]
            )
            .collect()
        )
        thresholds_long = (
            thresholds_wide.select([self.level] + [f"n{q}" for q in quantile_range])
            .melt(
                id_vars=self.level,
                value_vars=cs.numeric(),
                variable_name="Decile",
                value_name="Count",
            )
            .with_columns(pl.col("Decile").str.replace("n", "p"))
            .join(
                thresholds_wide.select(
                    [self.level] + [f"p{q}" for q in quantile_range]
                ).melt(
                    id_vars=self.level,
                    value_vars=cs.numeric(),
                    variable_name="Decile",
                    value_name="Threshold",
                ),
                on=[self.level, "Decile"],
            )
            .sort([self.level, "Threshold"])
        ).filter(pl.col(self.level) == "Arbitration")
        return thresholds_long

    def priority_component_distribution(self, component, granularity):
        distribution_data = (
            self.sample.filter(pl.col("Priority").is_not_null())
            .select([granularity, component])
            .sort(granularity)
        )
        return distribution_data

    def aggregate_remaining_per_stage(
        self, df: pl.LazyFrame, group_by_columns: List[str], aggregations: List = []
    ) -> pl.LazyFrame:
        """
        Workhorse function to convert the raw Decision Analyzer data (filter view) to
        the aggregates remaining per stage, ensuring all stages are represented.
        """
        stage_orders = (
            df.group_by(self.level)
            .agg(pl.min("StageOrder"))
            .with_columns(
                pl.col(self.level)
                .cast(pl.Utf8)
                .cast(
                    pl.Enum(self.AvailableNBADStages)
                )  # weird polars behaviour(version: 1.29), try removing in later patches
            )
        )

        def aggregate_over_remaining_stages(df, stage, remaining_stages):
            return (
                df.filter(pl.col(self.level).is_in(remaining_stages))
                .group_by(group_by_columns)
                .agg(aggregations)
                .with_columns(pl.lit(stage).alias(self.level))
            )

        aggs = {
            stage: aggregate_over_remaining_stages(
                df, stage, self.AvailableNBADStages[i:]
            )
            for (i, stage) in enumerate(self.AvailableNBADStages)
        }
        remaining_view = (
            pl.concat(aggs.values())
            .with_columns(
                pl.col(self.level).cast(stage_orders.collect_schema()[self.level])
            )
            .join(stage_orders, on=self.level, how="left")
        )

        return remaining_view

    # TODO refactor this into the DecisionData class and refactor to use the remaining view aggregator (aggregate_remaining_per_stage)

    def get_offer_quality(self, df, group_by):
        """
        Given a dataframe with filtered action counts at stages.
        Flips it to usual VF view by doing a rolling sum over stages.

        Parameters
        ----------
        df : pl.LazyFrame
            Decision Analyzer style filtered action counts dataframe.
        groupby_cols : list
            The list of column names to group by([self.level, "pxInteractionID"]).

        Returns
        -------
        pl.LazyFrame
            Value Finder style, available action counts per group_by category
        """
        dfs = []
        for i, stage in enumerate(self.AvailableNBADStages):
            stage_df = (
                # TODO refactor to use the remaining view aggregator (aggregate_remaining_per_stage)
                df.filter(pl.col(self.level).is_in(self.AvailableNBADStages[i:]))
                .group_by(group_by)
                .agg(
                    pl.sum(
                        "no_of_offers",
                        "new_models",
                        "poor_propensity_offers",
                        "poor_priority_offers",
                        "good_offers",
                    )
                )
                .with_columns(pl.lit(stage).alias(self.level))
            )
            dfs.append(stage_df)
        stage_df = pl.concat(dfs)
        stage_df = stage_df.with_columns(
            has_no_offers=pl.when(pl.col("no_of_offers") == 0)
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
            atleast_one_relevant_action=pl.when(pl.col("good_offers") >= 1)
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
            only_irrelevant_actions=pl.when(pl.col("good_offers") == 0)
            .then(pl.lit(1))
            .otherwise(0),
        )
        return stage_df

    @cached_property
    def get_overview_stats(self):
        """Creates an overview from sampled data"""

        nOffersPerStage = (
            self.get_optionality_data(self.sample)
            .group_by(self.level)
            .agg(pl.col("nOffers").mean().round().cast(pl.Int16))
            .collect()
        )

        def _offer_counts(stage):
            stage_values = nOffersPerStage[self.level].to_list()
            return (
                (
                    nOffersPerStage.filter(pl.col(self.level) == stage)
                    .select("nOffers")
                    .item()
                )
                if stage in stage_values
                else 0
            )

        kpis = (
            (
                self.getPreaggregatedFilterView.select(
                    pl.n_unique("pyName").alias("Actions"),
                    pl.n_unique("pyChannel").alias(
                        "Channels"
                    ),  # TODO plus direction of course
                    (
                        (pl.max("pxDecisionTime_max") - pl.min("pxDecisionTime_min"))
                        + pl.duration(days=1)
                    ).alias("Duration"),
                    pl.min("pxDecisionTime_min").cast(pl.Date).alias("StartDate"),
                )
            )
            .collect()
            .hstack(
                self.sample.select(
                    pl.n_unique("pySubjectID").alias("Customers"),
                    pl.n_unique("pxInteractionID").alias("Decisions"),
                ).collect()
            )
            .hstack(
                pl.DataFrame(
                    {
                        "avgOffersAtArbitration": _offer_counts("Arbitration"),
                        "avgAvailable": _offer_counts("AvailableActions"),
                    }
                )
            )
        )

        return {k: kpis[k].item() for k in kpis.columns}

    # TODO think about caching
    def get_sensitivity(self, win_rank=1, filters=None):
        """
        Global Sensitivity: Number of decisions where original rank-1 changes.
        Local Sensitivity: Number of times the selected offer(s) are in the rank-1 when dropping one of the prioritization factors.

        Parameters
        ----------
        win_rank: Int
            Maximum rank to be considered a winner.
        filters: List[pl.Expr]
            Selected offers, only used in local sensitivity analysis.

        Returns
        -------
        pl.LazyFrame
        """
        if filters is None:
            filters = pl.col("pxRank") <= win_rank

        sensitivity = (
            apply_filter(
                self.reRank(), filters
            )  # don't put filters in rerank function, we need to filter after reranking!
            # .filter(pl.col("rank_PVCL") <= win_rank)
            .select(
                [
                    pl.col("pxInteractionID")
                    .filter(pl.col(x) <= win_rank)
                    .n_unique()
                    .cast(
                        pl.Int32
                    )  # thinks they are unsigned int, 33-34 returns big number
                    .alias(
                        (f"{x.split('_')[1]}_win_count")
                    )  # calculating win_counts of different combinations
                    for x in [
                        "rank_PVCL",
                        "rank_VCL",
                        "rank_PCL",
                        "rank_PVL",
                        "rank_PVC",
                    ]
                ]
            )
            .with_columns(
                [
                    (pl.col("PVCL_win_count") - pl.col(x)).alias(x)
                    for x in [
                        "VCL_win_count",
                        "PCL_win_count",
                        "PVL_win_count",
                        "PVC_win_count",
                    ]
                ]
            )
            .rename(
                {
                    "PVCL_win_count": "Priority",
                    "VCL_win_count": "Propensity",
                    "PCL_win_count": "Value",
                    "PVL_win_count": "Context Weights",
                    "PVC_win_count": "Levers",
                }
            )
            .melt(variable_name="Factor", value_name="Influence")
        )
        return sensitivity

    def get_offer_variability_stats(self, stage):
        offer_variability_data = self.getActionVariationData(stage)
        return {
            "n90": bisect_left(
                offer_variability_data.select("DecisionsFraction").collect()[
                    "DecisionsFraction"
                ],
                0.9,
            )
            - 1,  # first one is dummy
            "gini": 1.0
            - gini_coefficient(
                offer_variability_data.select(
                    ["ActionsFraction", "DecisionsFraction"]
                ).collect(),
                "ActionsFraction",
                "DecisionsFraction",
            ),
        }

    def get_winning_or_losing_interactions(self, win_rank, group_filter, win: bool):
        if win:
            rank_filter = pl.col("pxRank") <= win_rank
        else:
            rank_filter = pl.col("pxRank") > win_rank
        return (
            apply_filter(self.sample, group_filter)
            .filter(
                rank_filter
                & (pl.col(self.level).is_in(self.stages_from_arbitration_down))
            )
            .select(pl.col("pxInteractionID").unique())
        )

    def winning_from(self, interactions, win_rank, groupby_cols, top_k):
        winning_from = (
            self.sample.filter(
                pl.col("pxRank") > win_rank
            )  # TODO generalize this to any stage from Arbitration up but excluding Final
            .join(
                interactions,
                on="pxInteractionID",
                how="inner",
            )
            .group_by(groupby_cols)
            .agg(Decisions=pl.len())
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
            .head(top_k)
        )
        return winning_from

    def losing_to(self, interactions, win_rank, groupby_cols, top_k):
        return (
            self.sample.filter(pl.col("pxRank") <= win_rank)
            .join(
                interactions,
                on="pxInteractionID",
                how="inner",
            )
            .group_by(groupby_cols)
            .agg(Decisions=pl.len())
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
            .head(top_k)
        )

    def get_win_distribution_data(
        self,
        lever_condition: pl.Expr,
        lever_value: Optional[float] = None,
        all_interactions: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Calculate win distribution data for business lever analysis.

        This method generates distribution data showing how actions perform in
        arbitration decisions, both in baseline conditions and optionally with
        lever adjustments applied.

        Parameters
        ----------
        lever_condition : pl.Expr
            Polars expression defining which actions to apply the lever to.
            Example: pl.col("pyName") == "SpecificAction" or
                    (pl.col("pyIssue") == "Service") & (pl.col("pyGroup") == "Cards")
        lever_value : float, optional
            The lever multiplier value to apply to selected actions.
            If None, returns baseline distribution only.
            If provided, returns both original and lever-adjusted win counts.
        all_interactions : int, optional
            Total number of interactions to calculate "no winner" count.
            If provided, enables calculation of interactions without any winner.
            If None, "no winner" data is not calculated.

        Returns
        -------
        pl.DataFrame
            DataFrame containing win distribution with columns:
            - pyIssue, pyGroup, pyName: Action identifiers
            - original_win_count: Number of rank-1 wins in baseline scenario
            - new_win_count: Number of rank-1 wins after lever adjustment (only if lever_value provided)
            - n_decisions_survived_to_arbitration: Number of arbitration decisions the action participated in
            - selected_action: "Selected" for actions matching lever_condition, "Rest" for others
            - no_winner_count: Number of interactions without any winner (only if all_interactions provided)

        Notes
        -----
        - Only includes actions that survive to arbitration stage
        - Win counts represent rank-1 (first place) finishes in arbitration decisions
        - This is a zero-sum analysis: boosting selected actions suppresses others
        - Results are sorted by win count (new_win_count if available, else original_win_count)
        - When all_interactions is provided, "no winner" represents interactions without any rank-1 winner

        Examples
        --------
        Get baseline distribution for a specific action:
        >>> lever_cond = pl.col("pyName") == "MyAction"
        >>> baseline = decision_analyzer.get_win_distribution_data(lever_cond)

        Get distribution with 2x lever applied to service actions:
        >>> lever_cond = pl.col("pyIssue") == "Service"
        >>> with_lever = decision_analyzer.get_win_distribution_data(lever_cond, 2.0)

        Get distribution with no winner count:
        >>> total_interactions = 10000
        >>> with_no_winner = decision_analyzer.get_win_distribution_data(lever_cond, 2.0, total_interactions)
        """
        if lever_value is None:
            # Return baseline distribution only
            original_winners = self.reRank(
                additional_filters=pl.col("StageGroup").is_in(
                    self.stages_from_arbitration_down
                ),
            ).select(["pyIssue", "pyGroup", "pyName"] + ["pxInteractionID", "pxRank"])

            result = (
                original_winners.group_by(["pyIssue", "pyGroup", "pyName"])
                .agg(
                    original_win_count=pl.col("pxRank")
                    .filter(pl.col("pxRank") == 1)
                    .len(),
                    n_decisions_survived_to_arbitration=pl.col(
                        "pxInteractionID"
                    ).n_unique(),
                )
                .with_columns(
                    selected_action=pl.when(lever_condition)
                    .then(pl.lit("Selected"))
                    .otherwise(pl.lit("Rest"))
                )
                .collect()
                .sort("original_win_count", descending=True)
            )

            # Add no winner count if all_interactions is provided
            if all_interactions is not None:
                interactions_with_winners = (
                    original_winners.select("pxInteractionID").collect().n_unique()
                )
                no_winner_count = max(
                    0, all_interactions - interactions_with_winners
                )  # Ensure non-negative

                # Create a row with the same data types as the result
                no_winner_data = {
                    "pyIssue": ["No Winner"],
                    "pyGroup": ["No Winner"],
                    "pyName": ["No Winner"],
                    "original_win_count": [no_winner_count],
                    "n_decisions_survived_to_arbitration": [0],
                    "selected_action": ["No Winner"],
                }

                # Cast to match result schema
                no_winner_row = pl.DataFrame(no_winner_data).cast(result.schema)

                result = pl.concat([result, no_winner_row])

            return result
        else:
            # Return both baseline and lever-adjusted distribution
            recalculated_winners = self.reRank(
                overrides=[
                    (
                        pl.when(lever_condition)
                        .then(pl.lit(lever_value))
                        .otherwise(pl.col("Levers"))
                    ).alias("Levers")
                ],
                additional_filters=pl.col("StageGroup").is_in(
                    self.stages_from_arbitration_down
                ),
            ).select(
                ["pyIssue", "pyGroup", "pyName"]
                + ["pxInteractionID", "pxRank", "rank_PVCL"]
            )

            result = (
                recalculated_winners.group_by(["pyIssue", "pyGroup", "pyName"])
                .agg(
                    original_win_count=pl.col("pxRank")
                    .filter(pl.col("pxRank") == 1)
                    .len(),
                    new_win_count=pl.col("rank_PVCL")
                    .filter(pl.col("rank_PVCL") == 1)
                    .len(),
                    n_decisions_survived_to_arbitration=pl.col(
                        "pxInteractionID"
                    ).n_unique(),
                )
                .with_columns(
                    selected_action=pl.when(lever_condition)
                    .then(pl.lit("Selected"))
                    .otherwise(pl.lit("Rest"))
                )
                .collect()
                .sort("new_win_count", descending=True)
            )

            # Add no winner count if all_interactions is provided
            if all_interactions is not None:
                # Calculate no winner count based on new ranking
                interactions_with_new_winners = (
                    recalculated_winners.filter(pl.col("rank_PVCL") == 1)
                    .select("pxInteractionID")
                    .collect()
                    .n_unique()
                )
                no_winner_count = max(
                    0, all_interactions - interactions_with_new_winners
                )  # Ensure non-negative

                # Create a row with the same data types as the result
                no_winner_data = {
                    "pyIssue": ["No Winner"],
                    "pyGroup": ["No Winner"],
                    "pyName": ["No Winner"],
                    "original_win_count": [0],  # No winner has no original wins
                    "new_win_count": [no_winner_count],
                    "n_decisions_survived_to_arbitration": [0],
                    "selected_action": ["No Winner"],
                }

                # Cast to match result schema
                no_winner_row = pl.DataFrame(no_winner_data).cast(result.schema)

                result = pl.concat([result, no_winner_row])

            return result

    def find_lever_value(
        self,
        lever_condition: pl.Expr,
        target_win_percentage: float,
        win_rank: int = 1,
        low: float = 0,
        high: float = 100,
        precision: float = 0.01,
        ranking_stages: List[str] = None,
    ) -> float:
        """
        Binary search algorithm to find lever value needed to achieve a desired win percentage.

        Parameters
        ----------
        lever_condition : pl.Expr
            Polars expression that defines which actions should receive the lever
        target_win_percentage : float
            The desired win percentage (0-100)
        win_rank : int, default 1
            Consider actions winning if they rank <= this value
        low : float, default 0
            Lower bound for lever search range
        high : float, default 100
            Upper bound for lever search range
        precision : float, default 0.01
            Search precision - smaller values give more accurate results
        ranking_stages : List[str], optional
            List of stages to include in analysis. Defaults to ["Arbitration"]

        Returns
        -------
        float
            The lever value needed to achieve the target win percentage

        Raises
        ------
        ValueError
            If the target win percentage cannot be achieved within the search range
        """
        if ranking_stages is None:
            ranking_stages = ["Arbitration"]

        def _calculate_action_win_percentage(lever: float) -> float:
            """Calculate win percentage for a given lever value"""
            ranked_df = self.reRank(
                overrides=[
                    (
                        pl.when(lever_condition)
                        .then(pl.lit(lever))
                        .otherwise(pl.col("Levers"))
                    ).alias("Levers")
                ],
                additional_filters=pl.col(self.level).is_in(
                    self.stages_from_arbitration_down
                ),
            ).filter(pl.col("rank_PVCL") <= win_rank)

            selected_wins = (
                ranked_df.filter(lever_condition)
                .select("pxInteractionID")
                .collect()
                .height
            )
            selected_total = ranked_df.select("pxInteractionID").collect().height
            percentage = (selected_wins / selected_total) * 100
            return percentage

        beginning_high = high
        beginning_low = low

        # Check if target is achievable within bounds
        low_percentage = _calculate_action_win_percentage(beginning_low)
        high_percentage = _calculate_action_win_percentage(beginning_high)

        if target_win_percentage < low_percentage:
            raise ValueError(
                f"Target {target_win_percentage}% is too low. Even at lever {beginning_low}, your actions win in {low_percentage:.1f}% of interactions at arbitration."
                "You might have interactions where only your selected actions survive until arbitration. So they will win no matter what."
            )
        elif target_win_percentage > high_percentage:
            raise ValueError(
                f"Target {target_win_percentage}% is too high. Even at lever {beginning_high}, you only get {high_percentage:.1f}%. "
                f"You can increase the search range."
            )

        while high - low > precision:
            mid = (low + high) / 2

            current_win_percentage = _calculate_action_win_percentage(mid)

            if current_win_percentage < target_win_percentage:
                low = mid
            else:
                high = mid

        final_lever = (low + high) / 2
        return final_lever

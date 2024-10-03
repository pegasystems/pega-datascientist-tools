from functools import cached_property
from typing import List, Optional, Union

import numpy as np
import polars as pl
import polars.selectors as cs

from .data_read_utils import validate_columns
from .utils import (
    NBADScope_Mapping,
    apply_filter,
    determine_extract_type,
    get_table_definition,
    process,
    bisect_left,
    gini_coefficient,
)
from .plots import Plot


class DecisionData:
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
        "pxEngagementStage",
        "ModelPositives",
        "ModelEvidence",
    ]

    def __init__(self, raw_data: pl.LazyFrame):
        self.plot = Plot(self)
        # pxEngagement Stage present?
        self.extract_type = determine_extract_type(raw_data)
        # all columns are present?
        validate_columns(raw_data, get_table_definition(self.extract_type))
        # cast datatypes
        raw_data = process(df=raw_data, table=self.extract_type, raise_on_unknown=False)
        self.unfiltered_raw_decision_data = self.cleanup_raw_data(raw_data)
        self.resetGlobalDataFilters()
        # TODO subset against available fields in the data
        # TODO maybe we'll also need some aggregates per customer ID. Not certain, lets postpone, current dataset is not very representative.

        available_columns = set(self.unfiltered_raw_decision_data.columns)
        self.preaggregation_columns = {
            "pyIssue",
            "pyGroup",
            "pyName",
            # "pyTreatment", # should be in there dependent on what's in the data
            "pyChannel",
            "pyDirection",
            "pxComponentName",
            "ModelControlGroup",
            "day",
        }
        self.preaggregation_columns = self.preaggregation_columns.intersection(
            available_columns
        )

        self.max_win_rank = 5
        self.AvailableNBADStages = ["Arbitration"]

        if self.extract_type == "explainability_extract":
            columns_to_remove = {"pxComponentName"}
            self.preaggregation_columns -= columns_to_remove

            self.NBADStages_FilterView = self.AvailableNBADStages
            self.NBADStages_RemainingView = self.AvailableNBADStages
            self.NBADStages_Mapping = {
                "Arbitration": "Arbitration"
            }  # doesn't it have "Final" implicitly? for rank 1 or other threshold that user can define

        # TODO: subset this against the stages that are actually available in the data. This can be less
        # or more. There is work for dev in DSM and NBAD to tag the various stages. This should probably
        # start with a superset in the right order.
        # TODO: support human-friendly names, to show "Final" as "Presented" for example
        elif self.extract_type == "decision_analyzer":
            self.AvailableNBADStages = [
                "Eligibility",
                "Applicability",
                "Suitability",
            ] + self.AvailableNBADStages

            available_stages = (
                self.unfiltered_raw_decision_data.select(
                    pl.col("pxEngagementStage").unique()
                )  # note: this could be expensive - should move to file metadata
                .collect()
                .get_column("pxEngagementStage")
                .to_list()
            )
            self.AvailableNBADStages = [
                stage for stage in self.AvailableNBADStages if stage in available_stages
            ]
            self.NBADStages_RemainingView = ["Total Available"] + list(
                map(lambda x: "After " + x, self.AvailableNBADStages)
            )
            self.NBADStages_FilterView = self.AvailableNBADStages + ["Final"]
            # "Final" is a specific value in DA data

            self.NBADStages_Mapping = {
                j: self.NBADStages_RemainingView[i]
                for (i, j) in enumerate(self.NBADStages_FilterView)
            }

        self.plot = Plot(self)

    @cached_property
    def stages_from_arbitration_down(self):
        """
        All stages in the filter view starting at Arbitration. This initially
        will just be [Arbitration, Final] but as we get more stages in there
        may be more here.
        """
        return self.NBADStages_FilterView[
            self.NBADStages_FilterView.index("Arbitration") :
        ]

    @cached_property
    def arbitration_stage(self):
        return self.sample.filter(
            pl.col("pxEngagementStage").is_in(["Arbitration", "Final"])
        )

    def _invalidate_cached_properties(self):
        """Resets the properties of the class"""
        cls = type(self)
        cached = {
            attr
            for attr in list(self.__dict__.keys())
            if (descriptor := getattr(cls, attr, None))
            if isinstance(descriptor, cached_property)
        }
        for attr in cached:
            del self.__dict__[attr]

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
        stats_cols = ["pxDecisionTime", "Value", "FinalPropensity", "Priority"]
        exprs = [
            pl.col("pxInteractionID")
            .where(pl.col("pxRank") <= i)
            .count()
            .alias(f"Win_at_rank{i}")
            for i in range(1, self.max_win_rank + 1)
        ] + [
            pl.min(stats_cols).name.suffix("_min"),
            pl.max(stats_cols).name.suffix("_max"),
            pl.col("FinalPropensity", "Priority").sample(
                n=num_samples, with_replacement=True, shuffle=True
            ),
            pl.count().alias("Decisions"),
        ]

        self.preaggregated_decision_data_filterview = (
            self.decision_data.group_by(
                self.preaggregation_columns.union({"pxEngagementStage"})
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
        num_samples = 1  # TODO: when > 1 this breaks the .explode() I'm doing in some places(thresholding analysis e.g.), need to solve
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
                    # pl.col("FinalPropensity").sample(
                    #     n=num_samples, with_replacement=True, shuffle=True
                    # ),  # a list sample values - for distribution plots
                    pl.first("FinalPropensity"),
                    pl.min("FinalPropensity_min"),
                    pl.max("FinalPropensity_max"),
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
        def _sample_it(s: pl.Series) -> pl.Series:
            unique_ids = s.unique()  # should this be approximate?
            num_samples = min(len(unique_ids), 10000)
            sampled_ids = np.random.choice(unique_ids, num_samples, replace=False)
            return s.is_in(sampled_ids)

        df = (
            self.decision_data.with_columns(
                pl.col("pxInteractionID").map_batches(_sample_it).alias("_sample")
            )
            .filter(pl.col("_sample"))
            .drop("_sample")
            .collect()
            .lazy()
        )
        return df

    def getAvailableFieldsForFiltering(self, categoricalOnly=False):
        available_fields = []

        if categoricalOnly:
            schema_keys = self.decision_data.select(
                cs.string(include_categorical=True)
            ).schema.keys()

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

        if "day" not in df.columns:
            df = df.with_columns(day=pl.col("pxDecisionTime").dt.date())

        if self.extract_type == "explainability_extract":
            df = df.with_columns(pxEngagementStage=pl.lit("Arbitration"))

        preproc_df = df.with_columns(
            pxEngagementStage=pl.col("pxEngagementStage").cast(
                pl.Categorical(ordering="physical")
            ),
        ).with_columns(
            pl.col(pl.Categorical).exclude("pxEngagementStage").cast(pl.Utf8)
        )
        return preproc_df

    def getPossibleScopeValues(self):
        options = [
            col for col in NBADScope_Mapping.keys() if col in self.decision_data.columns
        ]
        return options

    def getPossibleStageValues(self):
        options = self.NBADStages_FilterView
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
            .filter(pl.col("pxEngagementStage") == stage)
            .group_by(["day"] + [grouping_levels] if trend else grouping_levels)
            .agg(pl.sum("Decisions"))
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
        )

        return distribution_data

    def getFunnelData(
        self, level, additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None
    ) -> pl.LazyFrame:
        funnelData = self.aggregate_remaining_per_stage(
            df=apply_filter(self.getPreaggregatedRemainingView, additional_filters),
            group_by_columns=[level],
            aggregations=pl.sum("Decisions").alias("count"),
        ).filter(pl.col("count") > 0)

        return funnelData

    def getFilterComponentData(
        self, top_n, additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None
    ) -> pl.DataFrame:
        stages_actions_df = (
            apply_filter(self.getPreaggregatedFilterView, additional_filters)
            .filter(pl.col("pxEngagementStage") != "Final")
            .group_by("pxEngagementStage", "pxComponentName")
            .agg(pl.sum("Decisions").alias("Filtered Decisions"))
            .collect()
        )
        result = pl.concat(
            pl.collect_all(
                [
                    x.lazy()
                    .top_k(top_n, by="Filtered Decisions")
                    .sort("Filtered Decisions", descending=False)
                    for x in stages_actions_df.partition_by("pxEngagementStage")
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
            pl.col(x)
            .rank(descending=True)
            .over(["pxInteractionID"])
            .cast(pl.Int16)
            .alias(f"rank_{x.split('_')[1]}")
            for x in ["prio_PVCL", "prio_VCL", "prio_PCL", "prio_PVL", "prio_PVC"]
        ]

        rank_df = (
            apply_filter(self.sample, additional_filters)
            .with_columns(overrides)
            # TODO generalize this to support stages from arbitration to final - there already is a function for that
            .filter(pl.col("pxEngagementStage").is_in(["Arbitration", "Final"]))
            .with_columns(
                prio_PVCL=(
                    pl.col("FinalPropensity")
                    * pl.col("Value")
                    * pl.col("ContextWeight")
                    * pl.col("Weight")
                ),
                prio_VCL=(pl.col("Value") * pl.col("ContextWeight") * pl.col("Weight")),
                prio_PCL=(
                    pl.col("FinalPropensity")
                    * pl.col("ContextWeight")
                    * pl.col("Weight")
                ),
                prio_PVL=(
                    pl.col("FinalPropensity") * pl.col("Value") * pl.col("Weight")
                ),
                prio_PVC=(
                    pl.col("FinalPropensity")
                    * pl.col("Value")
                    * pl.col("ContextWeight")
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
                pl.col("pxEngagementStage") == "Arbitration"
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

    @cached_property
    def get_optionality_data(self):
        """
        Finding the average number of actions per stage without trend analysis.
        We have to go back to the interaction level data, no way to
        use pre-aggregations unfortunately.
        """
        expr = [
            pl.count().alias("nOffers"),
            pl.col("FinalPropensity")
            .where(pl.col("FinalPropensity") < 0.5)
            .max()
            .alias("bestPropensity"),
        ]
        optionality_data = (
            self.aggregate_remaining_per_stage(
                df=self.sample,
                group_by_columns=["pxInteractionID"],
                aggregations=expr,
            )
            .group_by(["nOffers", "pxEngagementStage"])
            .agg(
                Interactions=pl.count(), AverageBestPropensity=pl.mean("bestPropensity")
            )
            .sort("nOffers", descending=True)
        )
        return optionality_data

    @cached_property
    def get_optionality_data_with_trend(self):
        """
        Finding the average number of actions per stage with trend analysis.
        We have to go back to the interaction level data, no way to
        use pre-aggregations unfortunately.
        """
        expr = [
            pl.count().alias("nOffers"),
            pl.col("FinalPropensity")
            .where(pl.col("FinalPropensity") < 0.5)
            .max()
            .alias("bestPropensity"),
        ]
        optionality_data = (
            self.aggregate_remaining_per_stage(
                df=self.sample,
                group_by_columns=["pxInteractionID", "day"],
                aggregations=expr,
            )
            .group_by(["nOffers", "pxEngagementStage", "day"])
            .agg(
                Interactions=pl.count(), AverageBestPropensity=pl.mean("bestPropensity")
            )
            .sort("nOffers", descending=True)
        )
        return optionality_data

    def getActionVariationData(self, stage):
        data = pl.concat(
            [
                pl.DataFrame(
                    {
                        "ActionIndex": 0,
                        "pyName": "",
                        "Decisions": 0,
                        # "pxEngagementStage": stage,
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
                ["pxEngagementStage", "ModelControlGroup"]
            )
            .agg(pl.count())
            .collect()
            .pivot(
                index="pxEngagementStage",
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
                pl.col("pxEngagementStage").is_in(self.stages_from_arbitration_down)
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
                + [pl.lit("Arbitration").alias("pxEngagementStage")]
            )
            .collect()
        )
        thresholds_long = (
            thresholds_wide.select(
                ["pxEngagementStage"] + [f"n{q}" for q in quantile_range]
            )
            .melt(
                id_vars="pxEngagementStage",
                value_vars=cs.numeric(),
                variable_name="Decile",
                value_name="Count",
            )
            .with_columns(pl.col("Decile").str.replace("n", "p"))
            .join(
                thresholds_wide.select(
                    ["pxEngagementStage"] + [f"p{q}" for q in quantile_range]
                ).melt(
                    id_vars="pxEngagementStage",
                    value_vars=cs.numeric(),
                    variable_name="Decile",
                    value_name="Threshold",
                ),
                on=["pxEngagementStage", "Decile"],
            )
            .sort(["pxEngagementStage", "Threshold"])
        ).filter(pl.col("pxEngagementStage") == "Arbitration")
        return thresholds_long

    def getValueDistributionData(self):
        ## TODO do we need this function? why don't we just pick arbitration and final from decisionData and select value?
        valueData = (
            self.getPreaggregatedFilterView.group_by(NBADScope_Mapping.keys())
            .agg(pl.min("Value_min"), pl.max("Value_max"))
            .sort(reversed(self.getPossibleScopeValues()))
        )  # Sort by action first
        return valueData

    def aggregate_remaining_per_stage(
        self, df: pl.LazyFrame, group_by_columns: List[str], aggregations: List = []
    ) -> pl.LazyFrame:
        """
        Workhorse function to convert the raw Decision Analyzer data (filter view) to
        the aggregates remaining per stage. Used all over the place.
        """

        def aggregate_over_remaining_stages(df, stage, remaining_stages):
            return (
                df.filter(pl.col("pxEngagementStage").is_in(remaining_stages))
                .group_by(group_by_columns)
                .agg(aggregations)
                .with_columns(pl.lit(stage).alias("pxEngagementStage"))
            )

        aggs = {
            stage: aggregate_over_remaining_stages(
                df, stage, self.NBADStages_FilterView[i:]
            )
            for (i, stage) in enumerate(self.NBADStages_FilterView)
        }
        remaining_view = pl.concat(aggs.values())
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
            The list of column names to group by(["pxEngagementStage", "pxInteractionID"]).

        Returns
        -------
        pl.LazyFrame
            Value Finder style, available action counts per group_by category
        """
        dfs = []
        for i, stage in enumerate(self.NBADStages_FilterView):
            stage_df = (
                # TODO refactor to use the remaining view aggregator (aggregate_remaining_per_stage)
                df.filter(
                    pl.col("pxEngagementStage").is_in(self.NBADStages_FilterView[i:])
                )
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
                .with_columns(pxEngagementStage=pl.lit(stage))
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
            self.get_optionality_data.group_by("pxEngagementStage")
            .agg(pl.col("nOffers").mean().round().cast(pl.Int16))
            .collect()
        )

        def _offer_counts(stage):
            return (
                (
                    nOffersPerStage.filter(pl.col("pxEngagementStage") == stage)
                    .select("nOffers")
                    .item()
                )
                if stage in nOffersPerStage["pxEngagementStage"]
                else 0
            )

        kpis = (
            (
                self.getPreaggregatedFilterView.select(
                    pl.approx_n_unique("pyName").alias("Actions"),
                    pl.approx_n_unique("pyChannel").alias(
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
                    pl.approx_n_unique("pySubjectID").alias("Customers"),
                    pl.approx_n_unique("pxInteractionID").alias("Decisions"),
                ).collect()
            )
            .hstack(
                pl.DataFrame(
                    {
                        "avgOffersAtArbitration": _offer_counts("Arbitration"),
                        "avgOffersAtEligibility": _offer_counts("Eligibility"),
                    }
                )
            )
        )

        return {k: kpis[k].item() for k in kpis.columns}

    # TODO think about caching
    def get_sensitivity(self, win_rank=1, filters=None):
        if filters is None:
            filters = pl.col("pxRank") <= win_rank

        global_sensitivity = (
            apply_filter(self.reRank(), filters)
            # .filter(pl.col("rank_PVCL") <= win_rank)
            .select(
                [
                    pl.col("pyName")
                    .where(pl.col(x) <= win_rank)
                    .count()
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
        return global_sensitivity

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
                & (pl.col("pxEngagementStage").is_in(self.stages_from_arbitration_down))
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
            .agg(Decisions=pl.count())
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
            .head(top_k)
        )
        return winning_from

    def losing_to(self, interactions, win_rank, groupby_cols, top_k):
        return (
            self.arbitration_stage.join(
                interactions,
                on="pxInteractionID",
                how="inner",
            )
            .group_by(groupby_cols)
            .agg(
                Actions=pl.col("pxRank")
                .where(pl.col("pxRank") <= win_rank.win_rank)
                .count()
            )
            .sort("Actions", descending=True)
            .filter(pl.col("Actions") > 0)
            .head(top_k)
        )

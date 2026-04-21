"""Aggregation namespace for :class:`DecisionAnalyzer`.

Methods in this module are exposed via ``da.aggregates.<method>``.  They
encapsulate the data-shaping queries that turn the pre-aggregated views
into the frames consumed by the plot layer and the Streamlit pages.
"""

from __future__ import annotations

__all__ = ["Aggregates"]

from bisect import bisect_left
from typing import Literal, TYPE_CHECKING

import polars as pl

from .utils import apply_filter, gini_coefficient

if TYPE_CHECKING:  # pragma: no cover
    from .DecisionAnalyzer import DecisionAnalyzer


class Aggregates:
    """Aggregation queries over the pre-aggregated views.

    Accessed via :attr:`DecisionAnalyzer.aggregates`.
    """

    def __init__(self, da: "DecisionAnalyzer") -> None:
        self.da = da

    def aggregate_remaining_per_stage(
        self,
        df: pl.LazyFrame,
        group_by_columns: list[str],
        aggregations: list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """
        Workhorse function to convert the raw Decision Analyzer data (filter view) to
        the aggregates remaining per stage, ensuring all stages are represented.
        """
        if aggregations is None:
            aggregations = []
        stage_orders = (
            df.group_by(self.da.level)
            .agg(pl.min("Stage Order"))
            .with_columns(
                pl.col(self.da.level)
                .cast(pl.Utf8)
                .cast(
                    pl.Enum(self.da.AvailableNBADStages)
                )  # weird polars behaviour(version: 1.29), try removing in later patches
            )
        )

        def aggregate_over_remaining_stages(df, stage, remaining_stages):
            return (
                df.filter(pl.col(self.da.level).is_in(remaining_stages))
                .group_by(group_by_columns)
                .agg(aggregations)
                .with_columns(pl.lit(stage).alias(self.da.level))
            )

        aggs = {
            stage: aggregate_over_remaining_stages(df, stage, self.da.AvailableNBADStages[i:])
            for (i, stage) in enumerate(self.da.AvailableNBADStages)
        }
        remaining_view = (
            pl.concat(aggs.values())
            .with_columns(pl.col(self.da.level).cast(stage_orders.collect_schema()[self.da.level]))
            .join(stage_orders, on=self.da.level, how="left")
        )

        return remaining_view

    def get_distribution_data(
        self,
        stage: str,
        grouping_levels: str | list[str],
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Distribution of decisions by grouping columns at a given stage.

        Parameters
        ----------
        stage : str
            Stage to filter on.
        grouping_levels : str or list of str
            Column(s) to group by (e.g. ``"Action"`` or ``["Channel", "Action"]``).
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters applied before aggregation.

        Returns
        -------
        pl.LazyFrame
            Columns from *grouping_levels* plus ``Decisions``, sorted descending.
        """
        distribution_data = (
            apply_filter(self.da.preaggregated_remaining_view, additional_filters)
            .filter(pl.col(self.da.level) == stage)
            .group_by(grouping_levels)
            .agg(pl.sum("Decisions"))
            .sort("Decisions", descending=True)
            .filter(pl.col("Decisions") > 0)
        )

        return distribution_data

    def get_funnel_data(
        self, scope: str, additional_filters: pl.Expr | list[pl.Expr] | None = None
    ) -> tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame]:
        filtered_df = apply_filter(self.da.preaggregated_filter_view, additional_filters)

        interaction_count_expr = (
            apply_filter(self.da.decision_data, additional_filters).select("Interaction ID").unique().count()
        )

        # Available: actions entering each stage (existing logic)
        available_data = self.aggregate_remaining_per_stage(
            df=filtered_df,
            group_by_columns=[scope],
            aggregations=[
                pl.sum("Decisions").alias("action_occurrences"),
                pl.col("Interaction_IDs")
                .list.explode(keep_nulls=False, empty_as_null=False)
                .unique()
                .count()
                .alias("interaction_count_for_scope"),
            ],
        ).filter(pl.col("action_occurrences") > 0)

        # Passing: actions exiting each stage (shift by one — use next stage's remaining set)
        # Output is the result, not a filtering stage, so exclude it from funnel views
        stages = [s for s in self.da.AvailableNBADStages if s != "Output"]
        # All stages including Output — needed to correctly count actions that survive
        # past the last filtering stage and reach Output.
        all_stages = list(self.da.AvailableNBADStages)

        def _passing_at_stage(i: int, stage: str) -> pl.DataFrame:
            # For stage i, passing = actions remaining at stage i+1 (including Output).
            all_stage_idx = all_stages.index(stage)
            next_stages = all_stages[all_stage_idx + 1 :]
            if not next_stages:
                next_stages = [stage]
            return (
                filtered_df.filter(pl.col(self.da.level).is_in(next_stages))
                .group_by([scope])
                .agg(
                    action_occurrences=pl.sum("Decisions"),
                    interaction_count_for_scope=pl.col("Interaction_IDs")
                    .list.explode(keep_nulls=False, empty_as_null=False)
                    .unique()
                    .count(),
                )
                .with_columns(pl.lit(stage).alias(self.da.level))
                .collect()
            )

        passing_data = pl.concat([_passing_at_stage(i, stage) for i, stage in enumerate(stages)]).filter(
            pl.col("action_occurrences") > 0
        )

        # Filtered: actions removed at each stage (existing logic, unchanged)
        filtered_funnel = (
            filtered_df.filter(pl.col("Record Type") == "FILTERED_OUT")
            .group_by([self.da.level, scope])
            .agg(
                action_occurrences=pl.sum("Decisions"),
                interaction_count_for_scope=pl.col("Interaction_IDs")
                .list.explode(keep_nulls=False, empty_as_null=False)
                .unique()
                .count(),
            )
            .collect()
        )

        interaction_count = interaction_count_expr.collect().item()
        metrics_expr = (
            pl.lit(interaction_count).alias("interaction_count"),
            (pl.col("action_occurrences") / pl.lit(interaction_count)).alias("actions_per_interaction"),
            ((pl.col("interaction_count_for_scope") / pl.lit(interaction_count)) * 100).alias("penetration_pct"),
        )

        return (
            available_data.with_columns(metrics_expr),
            passing_data.with_columns(metrics_expr),
            filtered_funnel.with_columns(metrics_expr),
        )

    def get_decisions_without_actions_data(
        self, additional_filters: pl.Expr | list[pl.Expr] | None = None
    ) -> pl.DataFrame:
        """Per-stage count of interactions newly left with no remaining actions.

        Returns a DataFrame with columns [self.da.level, "decisions_without_actions"],
        sorted in pipeline order. For each stage X, the value is the number of
        interactions that lose their final remaining action at stage X.
        """
        filter_view = apply_filter(self.da.preaggregated_filter_view, additional_filters)

        def count_from(stages):
            return (
                filter_view.filter(pl.col(self.da.level).is_in(stages))
                .select(pl.col("Interaction_IDs").list.explode(keep_nulls=False, empty_as_null=False).unique().count())
                .collect()
                .item()
            )

        total_interactions = (
            apply_filter(self.da.decision_data, additional_filters)
            .select("Interaction ID")
            .unique()
            .count()
            .collect()
            .item()
        )

        # Output is the result, not a filtering stage — exclude from this view
        stages = [s for s in self.da.AvailableNBADStages if s != "Output"]
        has_output = "Output" in self.da.AvailableNBADStages

        cumulative_without_actions: list[float] = []
        for i, _stage in enumerate(stages):
            successor_stages = list(stages[i + 1 :])
            # Include Output in every successor set so survivor sets remain nested,
            # even when some interactions skip optional intermediate stages.
            if has_output:
                successor_stages.append("Output")
            survivors_after_stage = count_from(successor_stages) if successor_stages else 0
            cumulative_without_actions.append(float(total_interactions - survivors_after_stage))

        rows = []
        for i, stage in enumerate(stages):
            prev_cumulative = cumulative_without_actions[i - 1] if i > 0 else 0.0
            rows.append(
                {
                    self.da.level: stage,
                    "decisions_without_actions": cumulative_without_actions[i] - prev_cumulative,
                }
            )

        return pl.DataFrame(rows)

    def get_funnel_summary(
        self,
        available_df: pl.LazyFrame,
        passing_df: pl.DataFrame,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Per-stage summary: Available, Passing, Filtered actions and Decisions.

        The table matches the funnel chart: it starts with a synthetic
        "Available Actions" row and excludes the Output stage.

        Parameters
        ----------
        available_df : pl.LazyFrame
            First element returned by ``get_funnel_data`` (actions entering each stage).
        passing_df : pl.DataFrame
            Second element returned by ``get_funnel_data`` (actions exiting each stage).
        additional_filters : optional
            Same filters used when calling ``get_funnel_data``.

        Returns
        -------
        pl.DataFrame
            One row per stage in pipeline order with raw counts first,
            then per-decision averages.
        """
        level = self.da.level
        stages = [s for s in self.da.AvailableNBADStages if s != "Output"]
        first_real_stage = stages[0] if stages else None

        available_by_stage = (
            available_df.collect()
            .with_columns(pl.col(level).cast(pl.Utf8))
            .group_by(level)
            .agg(pl.sum("action_occurrences").alias("Available Actions"))
        )
        passing_by_stage = (
            passing_df.with_columns(pl.col(level).cast(pl.Utf8))
            .group_by(level)
            .agg(pl.sum("action_occurrences").alias("Passing Actions"))
        )
        without_actions = self.get_decisions_without_actions_data(additional_filters)
        total_interactions = (
            apply_filter(self.da.decision_data, additional_filters)
            .select("Interaction ID")
            .unique()
            .count()
            .collect()
            .item()
        )
        stage_order_no_output = {s: i for i, s in enumerate(stages)}
        decisions_by_stage = (
            without_actions.with_columns(pl.col(level).cast(pl.Utf8))
            .sort(pl.col(level).replace_strict(stage_order_no_output, default=999, return_dtype=pl.Int32))
            .with_columns(pl.col("decisions_without_actions").cum_sum().alias("_cumulative_without_actions"))
            .with_columns((pl.lit(total_interactions) - pl.col("_cumulative_without_actions")).alias("Decisions"))
            .select([level, "Decisions"])
        )

        summary = (
            available_by_stage.join(passing_by_stage, on=level, how="left")
            .join(decisions_by_stage, on=level, how="left")
            .fill_null(0)
            .with_columns((pl.col("Available Actions") - pl.col("Passing Actions")).alias("Filtered Actions"))
        )

        # Exclude Output, keep only funnel stages
        summary = summary.filter(pl.col(level).is_in(stages))

        # Add synthetic "Available Actions" baseline row — but only when the first
        # real stage isn't already called "Available Actions" (to avoid a duplicate).
        synthetic_label = "Available Actions"
        if first_real_stage is not None and first_real_stage != synthetic_label:
            first_available = summary.filter(pl.col(level) == first_real_stage).select("Available Actions").item()
            synthetic_row = pl.DataFrame(
                {
                    level: [synthetic_label],
                    "Available Actions": [first_available],
                    "Passing Actions": [first_available],
                    "Decisions": [total_interactions],
                    "Filtered Actions": [0],
                }
            )
            summary = pl.concat([synthetic_row, summary], how="diagonal_relaxed")

        display_stage_order = [synthetic_label] + stages
        display_stage_index = {name: idx for idx, name in enumerate(display_stage_order)}
        summary = (
            summary.with_columns(
                (pl.col("Filtered Actions") / pl.col("Filtered Actions").sum() * 100)
                .round(1)
                .alias("% of Total Filtered"),
                (pl.col("Available Actions") / pl.lit(total_interactions)).round(2).alias("Avg Available/Decision"),
                (pl.col("Passing Actions") / pl.lit(total_interactions)).round(2).alias("Avg Passing/Decision"),
                (pl.col("Filtered Actions") / pl.lit(total_interactions)).round(2).alias("Avg Filtered/Decision"),
                ((pl.lit(total_interactions) - pl.col("Decisions")) / pl.lit(total_interactions) * 100)
                .round(1)
                .alias("% Decisions without Actions"),
            )
            .sort(pl.col(level).replace_strict(display_stage_index, default=999, return_dtype=pl.Int32))
            .select(
                level,
                "Available Actions",
                "Passing Actions",
                "Filtered Actions",
                "Decisions",
                "% of Total Filtered",
                "Avg Available/Decision",
                "Avg Passing/Decision",
                "Avg Filtered/Decision",
                "% Decisions without Actions",
            )
        )
        return summary

    def get_optionality_data(self, df: pl.LazyFrame | None = None, by_day: bool = False) -> pl.LazyFrame:
        """Average number of actions per stage, optionally broken down by day.

        Computes per-interaction action counts at each stage using
        ``aggregate_remaining_per_stage``, then aggregates into a histogram.

        Parameters
        ----------
        df : pl.LazyFrame, optional
            Input data.  Defaults to :attr:`sample`.
        by_day : bool, default False
            If True, include ``"day"`` in the grouping for trend analysis.
            When False, zero-action rows are injected for stages where some
            interactions have no remaining actions.

        Returns
        -------
        pl.LazyFrame
        """
        if df is None:
            df = self.da.sample
        expr = [
            pl.len().alias("nOffers"),
            pl.col("Propensity").filter(pl.col("Propensity") < 0.5).max().alias("bestPropensity"),
        ]

        group_by_columns = ["Interaction ID", "day"] if by_day else ["Interaction ID"]
        outer_group = ["nOffers", self.da.level, "day"] if by_day else ["nOffers", self.da.level]

        per_offer_count_and_stage = (
            self.aggregate_remaining_per_stage(
                df=df,
                group_by_columns=group_by_columns,
                aggregations=expr,
            )
            .group_by(outer_group)
            .agg(Interactions=pl.len(), AverageBestPropensity=pl.mean("bestPropensity"))
        )

        if by_day:
            return per_offer_count_and_stage.sort("nOffers", descending=True)

        # Non-trend mode: inject zero-action rows for completeness
        total_interactions = df.select("Interaction ID").collect().unique().height
        schema = per_offer_count_and_stage.collect_schema()
        zero_actions = (
            per_offer_count_and_stage.group_by(self.da.level)
            .agg(interaction_count=pl.sum("Interactions"))
            .with_columns(
                Interactions=(total_interactions - pl.col("interaction_count")).cast(schema["Interactions"]),
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

    def get_optionality_funnel(self, df: pl.LazyFrame | None = None) -> pl.LazyFrame:
        """Optionality funnel: interaction counts bucketed by available-action count.

        Buckets action counts into 0–6 and 7+, then counts interactions per
        stage and bucket.  Used by the optionality funnel chart.

        Parameters
        ----------
        df : pl.LazyFrame, optional
            Input data.  Defaults to :attr:`sample`.

        Returns
        -------
        pl.LazyFrame
        """
        if df is None:
            df = self.da.sample
        optionality_funnel = (
            self.get_optionality_data(df)
            .with_columns(
                pl.when(pl.col("nOffers") >= 7)
                .then(pl.lit("7+"))
                .otherwise(pl.col("nOffers").cast(str))
                .alias("available_actions")
            )
            .with_columns(
                pl.col("available_actions").cast(pl.Enum(["0", "1", "2", "3", "4", "5", "6", "7+"])),
            )
            .group_by([self.da.level, "available_actions"])
            .agg(pl.sum("Interactions"))
            .sort([self.da.level, "available_actions"])
        )
        return optionality_funnel

    def get_action_variation_data(
        self,
        stage: str,
        color_by: str | None = None,
    ) -> pl.LazyFrame:
        """Get action variation data, optionally broken down by a categorical dimension.

        Args:
            stage: The stage to analyze
            color_by: Optional categorical column to break down the variation by.
                     Can use "Channel/Direction" to combine Channel and Direction columns.
        """
        # Handle combined Channel/Direction column
        needs_channel_direction = color_by == "Channel/Direction"
        if needs_channel_direction:
            grouping_levels = ["Channel", "Direction", "Action"]
        else:
            grouping_levels = ["Action"] if color_by is None else [color_by, "Action"]

        # Get distribution data grouped by Action (and optionally color_by dimension)
        distribution = self.get_distribution_data(
            stage=stage,
            grouping_levels=grouping_levels,
        ).collect()

        # Create combined Channel/Direction column if needed
        if needs_channel_direction:
            distribution = distribution.with_columns(
                pl.concat_str("Channel", "Direction", separator="/").alias("Channel/Direction")
            )
            color_by_col = "Channel/Direction"
        else:
            color_by_col = color_by

        # Compute cumulative stats within each group if color_by is specified
        if color_by is not None:
            # Define final columns to keep
            final_columns = [
                "ActionIndex",
                "Action",
                "Decisions",
                "cumDecisions",
                "DecisionsFraction",
                "ActionsFraction",
                color_by_col,
            ]

            data = (
                distribution.sort([color_by_col, "Decisions"], descending=[False, True])
                .with_columns(cumDecisions=pl.col("Decisions").cum_sum().over(color_by_col).cast(pl.Int32))
                .with_columns(
                    DecisionsFraction=(pl.col("cumDecisions") / pl.col("Decisions").sum().over(color_by_col)),
                    Decisions=pl.col("Decisions").cast(pl.Int32),
                )
                .with_columns(ActionIndex=pl.int_range(pl.len()).over(color_by_col) + 1)
                .with_columns(
                    ActionsFraction=pl.col("ActionIndex") / pl.col("ActionIndex").max().over(color_by_col),
                    ActionIndex=pl.col("ActionIndex").cast(pl.UInt32),
                    Decisions=pl.col("Decisions").cast(pl.UInt32),
                    cumDecisions=pl.col("cumDecisions").cast(pl.UInt32),
                )
                .select(final_columns)
            )

            # Create zero points for each unique group value
            unique_groups = data[color_by_col].unique()
            zero_points = (
                pl.DataFrame(
                    {
                        color_by_col: unique_groups,
                        "ActionIndex": [0] * len(unique_groups),
                        "Action": [""] * len(unique_groups),
                        "Decisions": [0] * len(unique_groups),
                        "cumDecisions": [0] * len(unique_groups),
                        "DecisionsFraction": [0.0] * len(unique_groups),
                        "ActionsFraction": [0.0] * len(unique_groups),
                    }
                )
                .with_columns(
                    pl.col("ActionIndex").cast(pl.UInt32),
                    pl.col("Decisions").cast(pl.UInt32),
                    pl.col("cumDecisions").cast(pl.UInt32),
                )
                .select(final_columns)
            )

            result = pl.concat([zero_points, data]).sort([color_by_col, "ActionIndex"])
        else:
            data = (
                distribution.with_columns(cumDecisions=pl.col("Decisions").cum_sum().cast(pl.Int32))
                .with_columns(
                    DecisionsFraction=(pl.col("cumDecisions") / pl.sum("Decisions")),
                    Decisions=pl.col("Decisions").cast(pl.Int32),
                )
                .with_row_index("ActionIndex", 1)
                .with_columns(
                    ActionsFraction=pl.col("ActionIndex") / pl.len(),
                    ActionIndex=pl.col("ActionIndex").cast(pl.UInt32),
                    Decisions=pl.col("Decisions").cast(pl.UInt32),
                    cumDecisions=pl.col("cumDecisions").cast(pl.UInt32),
                )
            )

            # Single zero point for non-grouped data
            zero_point = pl.DataFrame(
                {
                    "ActionIndex": 0,
                    "Action": "",
                    "Decisions": 0,
                    "cumDecisions": 0,
                    "DecisionsFraction": 0.0,
                    "ActionsFraction": 0.0,
                }
            ).with_columns(
                pl.col("ActionIndex").cast(pl.UInt32),
                pl.col("Decisions").cast(pl.UInt32),
                pl.col("cumDecisions").cast(pl.UInt32),
            )

            result = pl.concat([zero_point, data])

        return result.lazy()

    def get_offer_variability_stats(self, stage: str) -> dict[str, float]:
        """Summary statistics for action variation at a stage.

        Parameters
        ----------
        stage : str
            Stage to analyse.

        Returns
        -------
        dict
            ``n90`` — number of actions covering 90 % of decisions.
            ``gini`` — Gini coefficient of decision concentration.
        """
        offer_variability_data = self.get_action_variation_data(stage)
        return {
            "n90": bisect_left(
                offer_variability_data.select("DecisionsFraction").collect()["DecisionsFraction"],
                0.9,
            )
            - 1,  # first one is dummy
            "gini": 1.0
            - gini_coefficient(
                offer_variability_data.select(["ActionsFraction", "DecisionsFraction"]).collect(),
                "ActionsFraction",
                "DecisionsFraction",
            ),
        }

    def get_offer_quality(self, df: pl.LazyFrame, group_by: str | list[str]) -> pl.LazyFrame:
        """Cumulative offer-quality breakdown across stages.

        Takes a filtered-action-counts frame (from :meth:`filtered_action_counts`)
        and converts it to a remaining-per-stage view, joining in customers
        that have zero actions so they are counted as well.

        Parameters
        ----------
        df : pl.LazyFrame
            Filtered action counts with columns ``no_of_offers``,
            ``new_models``, ``poor_propensity_offers``, etc.
        group_by : str or list of str
            Columns to group by (e.g. ``["Interaction ID"]``).

        Returns
        -------
        pl.LazyFrame
            Per-stage quality classification with boolean flag columns
            (``has_no_offers``, ``atleast_one_relevant_action``, etc.).
        """
        # Get all unique customers/interactions from the sample to ensure we count those with no actions
        if isinstance(group_by, str):
            group_by = [group_by]

        # Get the full set of customers from the FIRST stage (where all customers start)
        # This ensures we track all customers throughout the pipeline
        first_stage = self.da.AvailableNBADStages[0]
        all_customers = (
            self.da.sample.filter(pl.col(self.da.level) == first_stage).select(group_by).unique().collect().lazy()
        )

        dfs = []
        for i, stage in enumerate(self.da.AvailableNBADStages):
            # Get aggregated counts for this stage (only interactions WITH actions)
            stage_df = (
                # Tracked in docs/plans/decision-analyzer-TODO.md (P2:
                # Refactor get_offer_quality) — delegate to
                # aggregate_remaining_per_stage instead of this manual loop.
                df.filter(pl.col(self.da.level).is_in(self.da.AvailableNBADStages[i:]))
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
                .with_columns(pl.lit(stage).alias(self.da.level))
            )

            # Left join with all customers to include those with no actions (fill with 0s)
            # Cast stage_df level column to categorical to match join keys
            stage_df = stage_df.with_columns(pl.col(self.da.level).cast(pl.Categorical))

            # Add stage to all customers and join
            stage_customers = all_customers.with_columns(pl.lit(stage).cast(pl.Categorical).alias(self.da.level))
            stage_df = stage_customers.join(stage_df, on=group_by + [self.da.level], how="left").with_columns(
                pl.col(
                    "no_of_offers", "new_models", "poor_propensity_offers", "poor_priority_offers", "good_offers"
                ).fill_null(0)
            )

            dfs.append(stage_df)
        stage_df = pl.concat(dfs)

        # Determine if stage has propensity scores available (ActionPropensity and later)
        has_propensity = pl.col(self.da.level).is_in(self.da.stages_with_propensity)

        stage_df = stage_df.with_columns(
            has_no_offers=pl.when(pl.col("no_of_offers") == 0).then(pl.lit(1)).otherwise(pl.lit(0)),
            # For stages WITH propensity: classify as relevant (green) or irrelevant (orange)
            atleast_one_relevant_action=pl.when(has_propensity & (pl.col("good_offers") >= 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
            only_irrelevant_actions=pl.when(
                has_propensity & (pl.col("good_offers") == 0) & (pl.col("no_of_offers") > 0)
            )
            .then(pl.lit(1))
            .otherwise(0),
            # For stages WITHOUT propensity: just show as having actions (blue)
            atleast_one_action=pl.when(~has_propensity & (pl.col("no_of_offers") > 0))
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
        )
        return stage_df

    def filtered_action_counts(
        self,
        groupby_cols: list[str],
        propensity_th: float | None = None,
        priority_th: float | None = None,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Return action counts from the sample, optionally classified by propensity/priority thresholds.

        Parameters
        ----------
        groupby_cols : list of str
            Column names to group by.
        propensity_th : float, optional
            Propensity threshold for classifying offers.
        priority_th : float, optional
            Priority threshold for classifying offers.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample (e.g. channel filter).

        Returns
        -------
        pl.LazyFrame
            Aggregated action counts per group, with quality buckets when
            both thresholds are provided.
        """
        df = apply_filter(self.da.sample, additional_filters)
        additional_cols = ["Action", "Propensity"]
        required_cols = list(set(groupby_cols + additional_cols))
        for col in required_cols:
            if col not in df.collect_schema().names():
                raise ValueError(f"Column '{col}' not found in the dataframe.")

        if propensity_th is None or priority_th is None:
            return df.group_by(groupby_cols).agg(
                no_of_offers=pl.count("Action"),
            )

        # Only classify actions by propensity/priority at stages where propensity is available
        # Actions filtered before ActionPropensity stage don't have meaningful propensity scores
        has_propensity = pl.col(self.da.level).is_in(self.da.stages_with_propensity)

        propensity_classifying_expr = [
            pl.col("Action")
            .filter(has_propensity & (pl.col("Propensity") == 0.5) & (pl.col(self.da.level) != "Output"))
            .count()
            .alias("new_models"),
            pl.col("Action")
            .filter(has_propensity & (pl.col("Propensity") < propensity_th) & (pl.col("Propensity") != 0.5))
            .count()
            .alias("poor_propensity_offers"),
            pl.col("Action")
            .filter(has_propensity & (pl.col("Priority") < priority_th) & (pl.col("Propensity") != 0.5))
            .count()
            .alias("poor_priority_offers"),
            pl.col("Action")
            .filter(
                has_propensity
                & (pl.col("Propensity") >= propensity_th)
                & (pl.col("Propensity") != 0.5)
                & (pl.col("Priority") >= priority_th)
            )
            .count()
            .alias("good_offers"),
        ]
        return df.group_by(groupby_cols).agg(no_of_offers=pl.count("Action"), *propensity_classifying_expr)

    def get_trend_data(
        self,
        stage: str = "AvailableActions",
        scope: Literal["Group", "Issue", "Action"] | None = "Group",
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Daily trend of unique decisions from a given stage onward.

        Parameters
        ----------
        stage : str, default "AvailableActions"
            Starting stage; all stages from this point onward are included.
        scope : {"Group", "Issue", "Action"} or None, default "Group"
            Optional grouping dimension.  If ``None``, returns totals by day.
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters applied to the sample.

        Returns
        -------
        pl.DataFrame
            Columns: ``day``, optionally *scope*, and ``Decisions``.
        """
        stages = self.da.AvailableNBADStages[self.da.AvailableNBADStages.index(stage) :]
        group_by = ["day"] if scope is None else ["day", scope]

        trend_data = (
            apply_filter(self.da.sample, additional_filters)
            .filter(pl.col(self.da.level).is_in(stages))
            .group_by(group_by)
            .agg(pl.n_unique("Interaction ID").alias("Decisions"))
            .sort(group_by)
        )

        return trend_data

    def get_filter_component_data(
        self, top_n: int, additional_filters: pl.Expr | list[pl.Expr] | None = None
    ) -> pl.DataFrame:
        """Top-N filter components per stage, ranked by filtered-decision count.

        Parameters
        ----------
        top_n : int
            Maximum number of components to return per stage.
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters applied before aggregation.

        Returns
        -------
        pl.DataFrame
            Columns include the stage level, Component Name, and
            Filtered Decisions.
        """
        group_cols = [self.da.level, "Component Name"]
        available = set(self.da.preaggregated_filter_view.collect_schema().names())
        if "Component Type" in available:
            group_cols.append("Component Type")

        stages_actions_df = (
            apply_filter(self.da.preaggregated_filter_view, additional_filters)
            .filter(pl.col(self.da.level) != "Output")
            .group_by(group_cols)
            .agg(
                pl.col("Interaction_IDs")
                .list.explode(keep_nulls=False, empty_as_null=False)
                .unique()
                .count()
                .alias("Filtered Decisions")
            )
            .collect()
        )
        result = pl.concat(
            pl.collect_all(
                [
                    x.lazy().top_k(top_n, by="Filtered Decisions").sort("Filtered Decisions", descending=False)
                    for x in stages_actions_df.partition_by(self.da.level)
                ]
            )
        ).with_columns(pl.col(pl.Categorical).cast(pl.Utf8))

        return result

    def get_component_action_impact(
        self,
        top_n: int = 10,
        scope: str = "Action",
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Per-component breakdown of which items are filtered and how many.

        For each component, returns the top-N items (at the chosen scope
        granularity) it filters out. The scope controls whether the breakdown
        is at Issue, Group, or Action level.

        Parameters
        ----------
        top_n : int, default 10
            Maximum number of items to return per component.
        scope : str, default "Action"
            Granularity level: ``"Issue"``, ``"Group"``, or ``"Action"``.
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters to apply before aggregation.

        Returns
        -------
        pl.DataFrame
            Columns include Component Name, StageGroup, scope columns, and
            Filtered Decisions. Sorted by component then descending count.
        """
        filtered_data = apply_filter(self.da.decision_data, additional_filters).filter(
            pl.col("Record Type") == "FILTERED_OUT"
        )

        # Build scope columns up to and including the requested level
        scope_hierarchy = ["Issue", "Group", "Action"]
        scope_idx = scope_hierarchy.index(scope) if scope in scope_hierarchy else 2
        scope_cols = scope_hierarchy[: scope_idx + 1]

        group_cols = ["Component Name", self.da.level] + scope_cols
        available = set(filtered_data.collect_schema().names())
        group_cols = [c for c in group_cols if c in available]

        impact_df = filtered_data.group_by(group_cols).agg(pl.len().alias("Filtered Decisions")).collect()

        # Top-N items per component
        result = pl.concat(
            pl.collect_all(
                [
                    part.lazy().top_k(top_n, by="Filtered Decisions").sort("Filtered Decisions", descending=True)
                    for part in impact_df.partition_by("Component Name")
                ]
            )
        ).with_columns(pl.col(pl.Categorical).cast(pl.Utf8))

        return result.sort(["Component Name", "Filtered Decisions"], descending=[False, True])

    def get_component_drilldown(
        self,
        component_name: str,
        scope: str = "Action",
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
        sort_by: str = "Filtered Decisions",
    ) -> pl.DataFrame:
        """Deep-dive into a single filter component showing dropped actions and
        their potential value.

        Since scoring columns (Priority, Value, Propensity) are typically null
        on FILTERED_OUT rows, this method derives the action's "potential value"
        by looking up average scores from rows where the same action survives
        (non-null Priority/Value). This gives the "value of what's being
        dropped" perspective.

        Parameters
        ----------
        component_name : str
            The Component Name to drill into.
        scope : str, default "Action"
            Granularity level: ``"Issue"``, ``"Group"``, or ``"Action"``.
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters to apply before aggregation.
        sort_by : str, default "Filtered Decisions"
            Column to sort results by (descending).

        Returns
        -------
        pl.DataFrame
            Columns include scope columns, Filtered Decisions,
            avg_Priority, avg_Value, avg_Propensity, Component Type (if
            available).
        """
        base = apply_filter(self.da.decision_data, additional_filters)
        available = set(base.collect_schema().names())

        # Filtered rows for this component
        filtered_rows = base.filter(
            (pl.col("Record Type") == "FILTERED_OUT") & (pl.col("Component Name") == component_name)
        )

        scope_hierarchy = ["Issue", "Group", "Action"]
        scope_idx = scope_hierarchy.index(scope) if scope in scope_hierarchy else 2
        scope_cols = scope_hierarchy[: scope_idx + 1]

        group_cols = [c for c in scope_cols if c in available]
        agg_exprs = [pl.len().alias("Filtered Decisions")]
        if "Component Type" in available:
            group_cols.append("Component Type")

        filtered_agg = filtered_rows.group_by(group_cols).agg(agg_exprs).collect()

        # Reference scores from surviving rows (non-null Priority)
        score_cols = ["Priority", "Value", "Propensity"]
        present_scores = [c for c in score_cols if c in available]

        if present_scores:
            score_aggs = [pl.col(c).mean().alias(f"avg_{c}") for c in present_scores]
            join_cols = [c for c in scope_cols if c in available]
            reference_scores = (
                base.filter(pl.col("Priority").is_not_null()).group_by(join_cols).agg(score_aggs).collect()
            )
            result = filtered_agg.join(
                reference_scores,
                on=join_cols,
                how="left",
            )
        else:
            result = filtered_agg

        sort_col = sort_by if sort_by in result.columns else "Filtered Decisions"
        return result.with_columns(pl.col(pl.Categorical).cast(pl.Utf8)).sort(sort_col, descending=True)

    def get_ab_test_results(self) -> pl.DataFrame:
        """A/B test summary: control vs test counts and control percentage per stage.

        Returns
        -------
        pl.DataFrame
            One row per stage with columns for Control, Test counts and
            Control Percentage. Rows preserve the canonical
            ``AvailableNBADStages`` order (stages absent from the data are
            omitted).
        """
        tbl = (
            self.da.preaggregated_remaining_view.group_by(
                # Tracked in docs/plans/decision-analyzer-TODO.md (P3: AB
                # test: include IA properties) — IA properties not joined
                # here when present.
                [self.da.level, "Model Control Group"]
            )
            .agg(pl.len())
            .collect()
            .pivot(
                index=self.da.level,
                values="len",
                columns=["Model Control Group"],
                sort_columns=True,
            )
            .with_columns((pl.col("Control") / (pl.col("Test") + pl.col("Control"))).alias("Control Percentage"))
        )
        stage_order = pl.DataFrame(
            {self.da.level: self.da.AvailableNBADStages, "_stage_order": list(range(len(self.da.AvailableNBADStages)))}
        )
        tbl = (
            tbl.join(stage_order, on=self.da.level, how="left")
            .sort("_stage_order", nulls_last=True)
            .drop("_stage_order")
        )
        return tbl

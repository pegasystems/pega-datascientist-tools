"""Scoring/ranking/sensitivity namespace for :class:`DecisionAnalyzer`.

Methods in this module are exposed via ``da.scoring.<method>``.  They
cover re-ranking the priority formula, win/loss analysis, sensitivity
of the PVCL components, lever search, and quantile-based thresholding.
"""

from __future__ import annotations

__all__ = ["Scoring"]

from typing import Literal, TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from .utils import apply_filter

if TYPE_CHECKING:
    from .DecisionAnalyzer import DecisionAnalyzer


class Scoring:
    """Scoring, ranking and lever-analysis queries.

    Accessed via :attr:`DecisionAnalyzer.scoring`.
    """

    def __init__(self, da: "DecisionAnalyzer") -> None:
        self.da = da

    def re_rank(
        self,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
        overrides: list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Recalculate priority and rank for all PVCL component combinations.

        Computes five alternative priority scores by selectively dropping one
        component at a time (Propensity, Value, Context Weight, Levers) and
        ranks actions within each interaction for each variant.  This is the
        foundation for sensitivity analysis.

        Parameters
        ----------
        additional_filters : pl.Expr or list of pl.Expr, optional
            Filters applied to the sample before ranking.
        overrides : list of pl.Expr, optional
            Column override expressions applied before priority calculation
            (e.g. to simulate lever adjustments).

        Returns
        -------
        pl.LazyFrame
            Sample data augmented with ``prio_*`` and ``rank_*`` columns
            for each PVCL variant.
        """
        rank_exprs = [
            pl.struct(
                [
                    "is_mandatory",
                    x,
                    "Stage Order",
                    pl.col("Issue").rank() * -1,
                    pl.col("Group").rank() * -1,
                    pl.col("Action").rank() * -1,
                ]
            )
            .rank(descending=True, method="ordinal", seed=1)
            .over(["Interaction ID"])
            .cast(pl.Int16)
            .alias(f"rank_{x.split('_')[1]}")
            for x in ["prio_PVCL", "prio_VCL", "prio_PCL", "prio_PVL", "prio_PVC"]
        ]

        # Fill missing PVCL components with 1.0 (neutral for multiplication).
        available = set(self.da.sample.collect_schema().names())
        pvcl_defaults = {
            "Propensity": 1.0,
            "Value": 1.0,
            "Context Weight": 1.0,
            "Levers": 1.0,
        }
        fill_exprs = []
        for col_name, default in pvcl_defaults.items():
            if col_name in available:
                fill_exprs.append(pl.col(col_name).fill_null(default))
            else:
                fill_exprs.append(pl.lit(default).alias(col_name))

        rank_df = (
            apply_filter(
                self.da.sample.with_columns(fill_exprs),
                additional_filters,
            )
            .with_columns(overrides if overrides is not None else [])
            .filter(pl.col("Priority").is_not_null())
            .with_columns(
                prio_PVCL=(pl.col("Propensity") * pl.col("Value") * pl.col("Context Weight") * pl.col("Levers")),
                prio_VCL=(pl.col("Value") * pl.col("Context Weight") * pl.col("Levers")),
                prio_PCL=(pl.col("Propensity") * pl.col("Context Weight") * pl.col("Levers")),
                prio_PVL=(pl.col("Propensity") * pl.col("Value") * pl.col("Levers")),
                prio_PVC=(pl.col("Propensity") * pl.col("Value") * pl.col("Context Weight")),
            )
            .with_columns(*rank_exprs)
        )

        return rank_df

    def get_selected_group_rank_boundaries(
        self,
        group_filter: pl.Expr | list[pl.Expr],
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Compute selected-group rank boundaries per interaction.

        For each interaction where the selected comparison group is present,
        returns the best (lowest) and worst (highest) rank observed for the
        selected rows in arbitration-relevant stages.
        """
        selected_rows = (
            apply_filter(apply_filter(self.da.sample, additional_filters), group_filter)
            .filter(pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down))
            .select(["Interaction ID", "Rank"])
        )

        return selected_rows.group_by("Interaction ID").agg(
            selected_group_best_rank=pl.col("Rank").min(),
            selected_group_worst_rank=pl.col("Rank").max(),
            selected_group_row_count=pl.len(),
        )

    def _remaining_at_stage(
        self,
        stage: str | None = None,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Return sample rows remaining at *stage*.

        Uses the ``aggregate_remaining_per_stage`` logic: rows whose stage
        order is >= the selected stage are "remaining" there.  If *stage*
        is None, falls back to rows with non-null Priority.
        """
        base = apply_filter(self.da.sample, additional_filters)
        if stage is None:
            return base.filter(pl.col("Priority").is_not_null())
        stage_idx = self.da.AvailableNBADStages.index(stage) if stage in self.da.AvailableNBADStages else 0
        remaining_stages = self.da.AvailableNBADStages[stage_idx:]
        return base.filter(pl.col(self.da.level).is_in(remaining_stages))

    def get_sensitivity(
        self,
        win_rank: int = 1,
        group_filter: pl.Expr | list[pl.Expr] | None = None,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Global or local sensitivity of the prioritization factors.

        Parameters
        ----------
        win_rank : int
            Maximum rank to be considered a winner.
        group_filter : pl.Expr, optional
            Selected offers, only used in local sensitivity analysis.
            When ``None`` (global), results are cached by ``win_rank``.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample before re-ranking
            (e.g. channel filter).  When set, caching is bypassed.

        Returns
        -------
        pl.LazyFrame
        """
        is_global_sensitivity = group_filter is None
        if is_global_sensitivity and additional_filters is None:
            if win_rank in self.da._sensitivity_cache:
                return self.da._sensitivity_cache[win_rank]
            group_filter = pl.col("Rank") <= win_rank
        elif is_global_sensitivity:
            group_filter = pl.col("Rank") <= win_rank

        sensitivity = (
            apply_filter(
                self.re_rank(additional_filters=additional_filters), group_filter
            )  # don't put filters in rerank function, we need to filter after reranking!
            # .filter(pl.col("rank_PVCL") <= win_rank)
            .select(
                [
                    pl.col("Interaction ID")
                    .filter(pl.col(x) <= win_rank)
                    .n_unique()
                    .cast(pl.Int32)  # thinks they are unsigned int, 33-34 returns big number
                    .alias((f"{x.split('_')[1]}_win_count"))  # calculating win_counts of different combinations
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
            .unpivot(variable_name="Factor", value_name="Influence")
        )
        if is_global_sensitivity:
            sensitivity = sensitivity.with_columns(Influence=pl.col("Influence").abs())
            if additional_filters is None:
                self.da._sensitivity_cache[win_rank] = sensitivity
        return sensitivity

    def get_thresholding_data(self, fld: str, quantile_range=range(10, 100, 10)) -> pl.DataFrame:
        """Quantile-based thresholding analysis at Arbitration.

        Computes counts and threshold values at each quantile for the given
        field (*fld*).  Results are cached per ``(fld, quantile_range)``.

        Parameters
        ----------
        fld : str
            Column name to compute quantiles for (e.g. ``"Propensity"``).
        quantile_range : range, default ``range(10, 100, 10)``
            Percentile breakpoints to compute.

        Returns
        -------
        pl.DataFrame
            Long-format table with columns ``Decile``, ``Count``,
            ``Threshold``, and the stage-level column.
        """
        cache_key = (fld, tuple(quantile_range))
        if cache_key in self.da._thresholding_cache:
            return self.da._thresholding_cache[cache_key]

        thresholds_wide = (
            # Note: runs on pre-aggregated data - maybe should be using _min/_max for filtering instead
            self.da.preaggregated_filter_view.filter(pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down))
            .select(
                # TODO can probably code this up more efficiently
                [pl.col(fld).explode().quantile(q / 100.0).alias(f"p{q}") for q in quantile_range]
                + [
                    ((pl.col(fld).explode()) < (pl.col(fld).explode().quantile(q / 100.0))).sum().alias(f"n{q}")
                    for q in quantile_range
                ]
                + [pl.lit("Arbitration").alias(self.da.level)]
            )
            .collect()
        )
        thresholds_long = (
            thresholds_wide.select([self.da.level] + [f"n{q}" for q in quantile_range])
            .unpivot(
                index=self.da.level,
                on=cs.numeric(),
                variable_name="Decile",
                value_name="Count",
            )
            .with_columns(pl.col("Decile").str.replace("n", "p"))
            .join(
                thresholds_wide.select([self.da.level] + [f"p{q}" for q in quantile_range]).unpivot(
                    index=self.da.level,
                    on=cs.numeric(),
                    variable_name="Decile",
                    value_name="Threshold",
                ),
                on=[self.da.level, "Decile"],
            )
            .sort([self.da.level, "Threshold"])
        ).filter(pl.col(self.da.level) == "Arbitration")
        self.da._thresholding_cache[cache_key] = thresholds_long
        return thresholds_long

    def priority_component_distribution(
        self,
        component: str,
        granularity: str,
        stage: str | None = None,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Data for a single component's distribution, grouped by granularity.

        Parameters
        ----------
        component : str
            Column name of the component to analyze.
        granularity : str
            Column to group by (e.g. "Issue", "Group", "Action").
        stage : str, optional
            Filter to actions remaining at this stage. If None, uses all
            rows with non-null Priority.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample (e.g. channel filter).
        """
        cols = [granularity, component]
        df = self._remaining_at_stage(stage, additional_filters=additional_filters)
        return df.select(cols).sort(granularity)

    def all_components_distribution(
        self,
        granularity: str,
        stage: str | None = None,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Data for the overview panel: all prioritization components at once.

        Parameters
        ----------
        granularity : str
            Column to group by.
        stage : str, optional
            Filter to actions remaining at this stage.
        additional_filters : pl.Expr or list[pl.Expr], optional
            Extra filters applied to the sample (e.g. channel filter).
        """
        from .utils import PRIO_COMPONENTS

        available = set(self.da.sample.collect_schema().names())
        cols = [c for c in PRIO_COMPONENTS if c in available]
        df = self._remaining_at_stage(stage, additional_filters=additional_filters)
        return df.select([granularity] + cols).sort(granularity)

    def get_win_loss_distribution_data(
        self,
        level: str | list[str],
        win_rank: int | None = None,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
        group_filter: pl.Expr | list[pl.Expr] | None = None,
        status: Literal["Wins", "Losses"] | None = None,
        top_k: int | None = None,
    ) -> pl.LazyFrame:
        """Win/loss distribution at a given scope level.

        Operates in two modes depending on whether *group_filter* is provided:

        * **Without group_filter** (rank-based): uses pre-aggregated data and
          a fixed *win_rank* threshold to split wins/losses.
        * **With group_filter** (group-based): uses sample data and
          per-interaction rank boundaries of the selected group to identify
          actions it beats (*Wins*) or loses to (*Losses*).

        Parameters
        ----------
        level : str or list of str
            Column(s) to group the distribution by (e.g. ``"Action"``).
        win_rank : int, optional
            Fixed rank threshold (required when *group_filter* is ``None``).
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters applied before aggregation.
        group_filter : pl.Expr or list of pl.Expr, optional
            Filter defining the comparison group.
        status : {"Wins", "Losses"}, optional
            Required when *group_filter* is provided.
        top_k : int, optional
            Limit the number of rows returned (group_filter mode only).

        Returns
        -------
        pl.LazyFrame
        """
        if group_filter is None:
            if win_rank is None:
                raise ValueError("win_rank must be provided when group_filter is None.")
            win_col = f"Win_at_rank{win_rank}"
            group_level_win_losses = (
                apply_filter(self.da.preaggregated_remaining_view, additional_filters)
                .filter(pl.col(self.da.level) == "Arbitration")
                .group_by(level)
                .agg(Wins=pl.sum(win_col), Decisions=pl.sum("Decisions"))
                .with_columns(Losses=pl.col("Decisions") - pl.col("Wins"))
                .with_columns(
                    Wins=pl.col("Wins") / pl.sum("Wins"),
                    Losses=pl.col("Losses") / pl.sum("Losses"),
                )
            )

            group_level_win_losses = group_level_win_losses.unpivot(
                index=level,
                on=["Wins", "Losses"],
                variable_name="Status",
                value_name="Percentage",
            ).sort(["Status", "Percentage"], descending=True)

            return group_level_win_losses

        if status not in {"Wins", "Losses"}:
            raise ValueError("When group_filter is provided, status must be either 'Wins' or 'Losses'.")

        selected_group_rank_boundaries = self.get_selected_group_rank_boundaries(
            group_filter=group_filter,
            additional_filters=additional_filters,
        )

        stage_filtered_data = apply_filter(self.da.sample, additional_filters).filter(
            pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down)
        )

        if status == "Wins":
            # Actions beaten by the selected group are ranked below the selected worst rank.
            comparison_rows = stage_filtered_data.join(
                selected_group_rank_boundaries,
                on="Interaction ID",
                how="inner",
            ).filter(pl.col("Rank") > pl.col("selected_group_worst_rank"))
        else:
            # Actions that beat the selected group are ranked above the selected best rank.
            comparison_rows = stage_filtered_data.join(
                selected_group_rank_boundaries,
                on="Interaction ID",
                how="inner",
            ).filter(pl.col("Rank") < pl.col("selected_group_best_rank"))

        n_interactions = comparison_rows.select(pl.n_unique("Interaction ID")).collect().item()

        distribution = (
            comparison_rows.group_by(level)
            .agg(Decisions=pl.len())
            .filter(pl.col("Decisions") > 0)
            .sort("Decisions", descending=True)
        )
        if top_k is not None:
            distribution = distribution.head(top_k)

        if n_interactions > 0:
            distribution = distribution.with_columns(
                (pl.col("Decisions") / pl.lit(n_interactions)).round(1).alias("Avg per Decision"),
            )
        else:
            distribution = distribution.with_columns(
                pl.lit(0.0).alias("Avg per Decision"),
            )

        return distribution

    def _winning_from(
        self,
        interactions: pl.LazyFrame,
        win_rank: int,
        groupby_cols: list[str],
        top_k: int = 20,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Actions beaten by the comparison group in its winning interactions.

        Parameters
        ----------
        interactions
            Interaction IDs where the comparison group wins.
        win_rank
            Rank threshold used to define "winning".
        groupby_cols
            Columns to group by (e.g. ``["Issue", "Group", "Action"]``).
        top_k
            Return only the top-k entries.
        additional_filters
            Optional extra filters (e.g. channel filter).
        """
        stage_filtered = apply_filter(self.da.sample, additional_filters).filter(
            pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down)
        )
        rows = stage_filtered.join(interactions, on="Interaction ID", how="inner").filter(pl.col("Rank") > win_rank)
        n_interactions = rows.select(pl.n_unique("Interaction ID")).collect().item()
        dist = (
            rows.group_by(groupby_cols)
            .agg(Decisions=pl.len())
            .filter(pl.col("Decisions") > 0)
            .sort("Decisions", descending=True)
            .head(top_k)
        )
        if n_interactions > 0:
            dist = dist.with_columns(
                (pl.col("Decisions") / pl.lit(n_interactions)).round(1).alias("Avg per Decision"),
            )
        else:
            dist = dist.with_columns(pl.lit(0.0).alias("Avg per Decision"))
        return dist

    def _losing_to(
        self,
        interactions: pl.LazyFrame,
        win_rank: int,
        groupby_cols: list[str],
        top_k: int = 20,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Actions that beat the comparison group in its losing interactions.

        Parameters
        ----------
        interactions
            Interaction IDs where the comparison group loses.
        win_rank
            Rank threshold used to define "winning".
        groupby_cols
            Columns to group by (e.g. ``["Issue", "Group", "Action"]``).
        top_k
            Return only the top-k entries.
        additional_filters
            Optional extra filters (e.g. channel filter).
        """
        stage_filtered = apply_filter(self.da.sample, additional_filters).filter(
            pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down)
        )
        rows = stage_filtered.join(interactions, on="Interaction ID", how="inner").filter(pl.col("Rank") < win_rank)
        n_interactions = rows.select(pl.n_unique("Interaction ID")).collect().item()
        dist = (
            rows.group_by(groupby_cols)
            .agg(Decisions=pl.len())
            .filter(pl.col("Decisions") > 0)
            .sort("Decisions", descending=True)
            .head(top_k)
        )
        if n_interactions > 0:
            dist = dist.with_columns(
                (pl.col("Decisions") / pl.lit(n_interactions)).round(1).alias("Avg per Decision"),
            )
        else:
            dist = dist.with_columns(pl.lit(0.0).alias("Avg per Decision"))
        return dist

    def get_winning_or_losing_interactions(
        self,
        group_filter: pl.Expr | list[pl.Expr],
        win: bool,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> pl.LazyFrame:
        """Interaction IDs where the comparison group wins or loses.

        Parameters
        ----------
        group_filter : pl.Expr or list of pl.Expr
            Filter defining the comparison group.
        win : bool
            If True, return interactions where the group wins (there are
            lower-ranked actions outside the group).  If False, return
            interactions where the group loses (there are higher-ranked
            actions outside the group).
        additional_filters : pl.Expr or list of pl.Expr, optional
            Extra filters (e.g. channel filter).

        Returns
        -------
        pl.LazyFrame
            Single-column frame of unique ``Interaction ID`` values.
        """
        selected_group_rank_boundaries = self.get_selected_group_rank_boundaries(
            group_filter=group_filter,
            additional_filters=additional_filters,
        )
        stage_filtered_data = apply_filter(self.da.sample, additional_filters).filter(
            pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down)
        )

        if win:
            interaction_filter = pl.col("Rank") > pl.col("selected_group_worst_rank")
        else:
            interaction_filter = pl.col("Rank") < pl.col("selected_group_best_rank")

        return (
            stage_filtered_data.join(selected_group_rank_boundaries, on="Interaction ID", how="inner")
            .filter(interaction_filter)
            .select(pl.col("Interaction ID").unique())
        )

    def get_win_loss_counts(
        self,
        group_filter: pl.Expr | list[pl.Expr],
        win_rank: int = 1,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
    ) -> dict[str, int]:
        """Count wins and losses for the comparison group at a given rank threshold.

        A **win** is an interaction where at least one member of the comparison
        group achieves a rank of *win_rank* or better (lower). A **loss** is
        any interaction where the group participates but none rank that high.

        Parameters
        ----------
        group_filter
            Filter expression(s) defining the comparison group.
        win_rank
            Rank threshold. The group "wins" when its best rank <= win_rank.
        additional_filters
            Optional extra filters (e.g. channel/direction).

        Returns
        -------
        dict with keys ``"wins"``, ``"losses"``, ``"total"``.
        """
        boundaries = self.get_selected_group_rank_boundaries(
            group_filter=group_filter,
            additional_filters=additional_filters,
        )
        counts = boundaries.select(
            wins=(pl.col("selected_group_best_rank") <= win_rank).sum(),
            losses=(pl.col("selected_group_best_rank") > win_rank).sum(),
            total=pl.len(),
        ).collect()

        row = counts.row(0, named=True)
        return {"wins": row["wins"], "losses": row["losses"], "total": row["total"]}

    def get_win_loss_distributions(
        self,
        interactions_win: pl.LazyFrame,
        interactions_loss: pl.LazyFrame,
        groupby_cols: list[str],
        top_k: int,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
        group_filter: pl.Expr | list[pl.Expr] | None = None,
        win_rank: int | None = None,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """Distribution of actions the comparison group wins from and loses to.

        Computes two aggregated distributions in a single pass over the
        stage-filtered data:

        * **winning_from**: actions ranked below the comparison group (i.e.
          actions it beats).
        * **losing_to**: actions ranked above the comparison group (i.e.
          actions that beat it).

        Either *group_filter* or *win_rank* must be provided to define the
        rank boundary.  When *group_filter* is given the boundary is the
        per-interaction best/worst rank of the selected group.  When only
        *win_rank* is given a fixed rank threshold is used instead.

        Parameters
        ----------
        interactions_win
            Interaction IDs where the comparison group wins (from
            ``get_winning_or_losing_interactions(win=True)``).
        interactions_loss
            Interaction IDs where the comparison group loses (from
            ``get_winning_or_losing_interactions(win=False)``).
        groupby_cols
            Columns to group by (e.g. ``["Action"]``).
        top_k
            Return only the top-k entries per distribution.
        additional_filters
            Optional extra filters (e.g. channel/direction).
        group_filter
            Filter expression(s) defining the comparison group.
        win_rank
            Fixed rank threshold. Required when *group_filter* is ``None``.

        Returns
        -------
        tuple of (winning_from, losing_to), both ``pl.LazyFrame`` with
        columns from *groupby_cols* plus ``Decisions``.
        """
        stage_filtered_data = apply_filter(self.da.sample, additional_filters).filter(
            pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down)
        )

        if group_filter is None:
            if win_rank is None:
                raise ValueError("win_rank must be provided when group_filter is None.")
            win_rows = stage_filtered_data.filter(pl.col("Rank") > win_rank)
            loss_rows = stage_filtered_data.filter(pl.col("Rank") < win_rank)
        else:
            boundaries = self.get_selected_group_rank_boundaries(
                group_filter=group_filter,
                additional_filters=additional_filters,
            )
            joined = stage_filtered_data.join(boundaries, on="Interaction ID", how="inner")
            win_rows = joined.filter(pl.col("Rank") > pl.col("selected_group_worst_rank"))
            loss_rows = joined.filter(pl.col("Rank") < pl.col("selected_group_best_rank"))

        def _aggregate(rows: pl.LazyFrame, interactions: pl.LazyFrame) -> pl.LazyFrame:
            return (
                rows.join(interactions, on="Interaction ID", how="inner")
                .group_by(groupby_cols)
                .agg(Decisions=pl.len())
                .sort("Decisions", descending=True)
                .filter(pl.col("Decisions") > 0)
                .head(top_k)
            )

        return _aggregate(win_rows, interactions_win), _aggregate(loss_rows, interactions_loss)

    def get_win_distribution_data(
        self,
        lever_condition: pl.Expr,
        lever_value: float | None = None,
        all_interactions: int | None = None,
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
            Example: pl.col("Action") == "SpecificAction" or
                    (pl.col("Issue") == "Service") & (pl.col("Group") == "Cards")
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
            - Issue, Group, Action: Action identifiers
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
        >>> lever_cond = pl.col("Action") == "MyAction"
        >>> baseline = decision_analyzer.get_win_distribution_data(lever_cond)

        Get distribution with 2x lever applied to service actions:
        >>> lever_cond = pl.col("Issue") == "Service"
        >>> with_lever = decision_analyzer.get_win_distribution_data(lever_cond, 2.0)

        Get distribution with no winner count:
        >>> total_interactions = 10000
        >>> with_no_winner = decision_analyzer.get_win_distribution_data(lever_cond, 2.0, total_interactions)
        """
        if lever_value is None:
            # Return baseline distribution only
            original_winners = self.re_rank(
                additional_filters=pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down),
            ).select(["Issue", "Group", "Action"] + ["Interaction ID", "Rank"])

            result = (
                original_winners.group_by(["Issue", "Group", "Action"])
                .agg(
                    original_win_count=pl.col("Rank").filter(pl.col("Rank") == 1).len(),
                    n_decisions_survived_to_arbitration=pl.col("Interaction ID").n_unique(),
                )
                .with_columns(
                    selected_action=pl.when(lever_condition).then(pl.lit("Selected")).otherwise(pl.lit("Rest"))
                )
                .collect()
                .sort("original_win_count", descending=True)
            )

            # Add no winner count if all_interactions is provided
            if all_interactions is not None:
                winner_df = original_winners.select("Interaction ID").collect()
                interactions_with_winners = winner_df.n_unique()
                no_winner_count = max(0, all_interactions - interactions_with_winners)  # Ensure non-negative

                # Create a row with the same data types as the result
                no_winner_data = {
                    "Issue": ["No Winner"],
                    "Group": ["No Winner"],
                    "Action": ["No Winner"],
                    "original_win_count": [no_winner_count],
                    "n_decisions_survived_to_arbitration": [0],
                    "selected_action": ["No Winner"],
                }

                # Cast to match result schema
                no_winner_row = pl.DataFrame(no_winner_data).cast({k: v for k, v in result.schema.items()})

                result = pl.concat([result, no_winner_row])

            return result
        else:
            # Return both baseline and lever-adjusted distribution
            recalculated_winners = self.re_rank(
                overrides=[
                    (pl.when(lever_condition).then(pl.lit(lever_value)).otherwise(pl.col("Levers"))).alias("Levers")
                ],
                additional_filters=pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down),
            ).select(["Issue", "Group", "Action"] + ["Interaction ID", "Rank", "rank_PVCL"])

            result_lf = (
                recalculated_winners.group_by(["Issue", "Group", "Action"])
                .agg(
                    original_win_count=pl.col("Rank").filter(pl.col("Rank") == 1).len(),
                    new_win_count=pl.col("rank_PVCL").filter(pl.col("rank_PVCL") == 1).len(),
                    n_decisions_survived_to_arbitration=pl.col("Interaction ID").n_unique(),
                )
                .with_columns(
                    selected_action=pl.when(lever_condition).then(pl.lit("Selected")).otherwise(pl.lit("Rest"))
                )
            )
            result = result_lf.collect().sort("new_win_count", descending=True)

            # Add no winner count if all_interactions is provided
            if all_interactions is not None:
                # Calculate no winner count based on new ranking
                new_winner_df = recalculated_winners.filter(pl.col("rank_PVCL") == 1).select("Interaction ID").collect()
                interactions_with_new_winners = new_winner_df.n_unique()
                no_winner_count = max(0, all_interactions - interactions_with_new_winners)  # Ensure non-negative

                # Create a row with the same data types as the result
                no_winner_data = {
                    "Issue": ["No Winner"],
                    "Group": ["No Winner"],
                    "Action": ["No Winner"],
                    "original_win_count": [0],  # No winner has no original wins
                    "new_win_count": [no_winner_count],
                    "n_decisions_survived_to_arbitration": [0],
                    "selected_action": ["No Winner"],
                }

                # Cast to match result schema
                no_winner_row = pl.DataFrame(no_winner_data).cast({k: v for k, v in result.schema.items()})

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
        ranking_stages: list[str] | None = None,
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
        ranking_stages : list[str], optional
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
            ranked_df = self.re_rank(
                overrides=[(pl.when(lever_condition).then(pl.lit(lever)).otherwise(pl.col("Levers"))).alias("Levers")],
                additional_filters=pl.col(self.da.level).is_in(self.da.stages_from_arbitration_down),
            ).filter(pl.col("rank_PVCL") <= win_rank)

            selected_wins_df = ranked_df.filter(lever_condition).select("Interaction ID").collect()
            selected_wins = selected_wins_df.height
            selected_total_df = ranked_df.select("Interaction ID").collect()
            selected_total = selected_total_df.height
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

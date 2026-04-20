"""Aggregation methods for Interaction History analysis."""

from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.pega_outcomes import get_openrate_labels as _get_openrate_labels
from ..utils.types import QUERY

if TYPE_CHECKING:
    from .IH import IH as IH_Class


class Aggregates(LazyNamespace):
    """Aggregation methods for Interaction History data.

    This class provides aggregation capabilities for summarizing customer
    interaction data. It is accessed through the `aggregates` attribute of an
    :class:`~pdstools.ih.IH.IH` instance.

    All aggregation methods support:
    - Grouping by dimensions via `by` parameter
    - Time bucketing via `every` parameter
    - Data filtering via `query` parameter

    Attributes
    ----------
    ih : IH
        Reference to the parent IH instance.

    See Also
    --------
    pdstools.ih.IH : Main analysis class.
    pdstools.ih.Plots : Visualization methods.

    Examples
    --------
    >>> ih = IH.from_ds_export("interaction_history.zip")
    >>> ih.aggregates.summary_success_rates(by="Channel").collect()
    >>> ih.aggregates.summary_outcomes(every="1w").collect()

    """

    def __init__(self, ih: "IH_Class"):
        """Initialize an Aggregates instance.

        Parameters
        ----------
        ih : IH
            The parent IH instance providing the data.

        """
        super().__init__()
        self.ih = ih

    def summarize_by_interaction(
        self,
        by: str | list[str] | pl.Expr | None = None,
        every: str | timedelta | None = None,
        query: QUERY | None = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Summarize outcomes per interaction.

        Groups data by interaction ID and determines the outcome for each
        interaction based on configured positive/negative outcome labels.

        Parameters
        ----------
        by : str, list[str], or pl.Expr, optional
            Grouping dimension(s). Default is None.
        every : str or timedelta, optional
            Time aggregation period (e.g., "1d", "1w"). Default is None.
        query : QUERY, optional
            Polars expression to filter data before aggregation.
        debug : bool, default False
            If True, include the Outcomes column (list of unique outcome values per interaction).
            If False, the Outcomes column is dropped from the results.

            This parameter affects the return value structure, not logging output.
            For debug logging, use logging.basicConfig(level=logging.DEBUG).

        Returns
        -------
        pl.LazyFrame
            Interaction-level data with columns:

            - **InteractionID**: Unique interaction identifier
            - **Interaction_Outcome_<metric>**: Boolean outcome per metric
            - **Propensity**: Propensity value from interaction
            - Plus any grouping columns

        Notes
        -----
        An interaction has a positive outcome for a metric if any outcome
        matches the positive labels. Otherwise, if any matches negative
        labels, the outcome is False. Otherwise it's null.

        See Also
        --------
        summary_success_rates : Aggregated success rates.

        Examples
        --------
        >>> ih.aggregates.summarize_by_interaction(by="Channel").collect()

        """
        if every is not None:
            source = self.ih.data.with_columns(pl.col.OutcomeTime.dt.truncate(every))
        else:
            source = self.ih.data

        if not isinstance(by, list):
            by = [by] if by is not None else []  # type: ignore[list-item]
        by_pl_exprs: list[str | pl.Expr] = [x for x in by if isinstance(x, pl.Expr)]
        by_non_pl_exprs = [x for x in by if not isinstance(x, pl.Expr)]
        group_by_clause = cdh_utils.safe_flatten_list(
            by_non_pl_exprs + (["OutcomeTime"] if every is not None else []),
            by_pl_exprs,
        )

        # When channel-aware outcome labels are available, pre-compute per-row
        # Engagement positive/negative flags before grouping. This avoids applying
        # global lists that are incorrect for mixed-channel datasets.
        channel_aware = getattr(self.ih, "outcome_labels_used", None)
        if channel_aware:
            pos_eng: pl.Expr = pl.lit(False)
            neg_eng: pl.Expr = pl.lit(False)
            for channel_dir, labels in channel_aware.items():
                parts = channel_dir.split("/", 1)
                ch_cond = (
                    (pl.col("Channel") == parts[0]) & (pl.col("Direction") == parts[1])
                    if len(parts) == 2
                    else (pl.col("Channel") == parts[0])
                )
                if labels.get("Accepts"):
                    pos_eng = pos_eng | (ch_cond & pl.col("Outcome").is_in(labels["Accepts"]))
                if labels.get("Impressions"):
                    neg_eng = neg_eng | (ch_cond & pl.col("Outcome").is_in(labels["Impressions"]))

            # OpenRate: only applicable to outbound channels.
            # TODO: reconsider null vs False for non-applicable channel/metric combinations
            pos_or: pl.Expr = pl.lit(False)
            neg_or: pl.Expr = pl.lit(False)
            for channel_dir, labels in channel_aware.items():
                or_labels = _get_openrate_labels(channel_dir)
                if or_labels is None:
                    continue  # inbound — both stay False → null after agg
                parts = channel_dir.split("/", 1)
                ch_cond = (
                    (pl.col("Channel") == parts[0]) & (pl.col("Direction") == parts[1])
                    if len(parts) == 2
                    else (pl.col("Channel") == parts[0])
                )
                pos_or = pos_or | (ch_cond & pl.col("Outcome").is_in(or_labels["positive"]))
                neg_or = neg_or | (ch_cond & pl.col("Outcome").is_in(or_labels["negative"]))

            source = source.with_columns(
                _IsPositiveEngagement=pos_eng,
                _IsNegativeEngagement=neg_eng,
                _IsPositiveOpenRate=pos_or,
                _IsNegativeOpenRate=neg_or,
            )

        engagement_agg: pl.Expr = (
            (
                pl.when(pl.col("_IsPositiveEngagement").any())
                .then(pl.lit(True))
                .when(pl.col("_IsNegativeEngagement").any())
                .then(pl.lit(False))
                .alias("Interaction_Outcome_Engagement")
            )
            if channel_aware
            else (
                pl.when(pl.col.Outcome.is_in(self.ih.positive_outcome_labels["Engagement"]).any())
                .then(pl.lit(True))
                .when(pl.col.Outcome.is_in(self.ih.negative_outcome_labels["Engagement"]).any())
                .then(pl.lit(False))
                .alias("Interaction_Outcome_Engagement")
            )
        )

        openrate_agg: pl.Expr = (
            (
                pl.when(pl.col("_IsPositiveOpenRate").any())
                .then(pl.lit(True))
                .when(pl.col("_IsNegativeOpenRate").any())
                .then(pl.lit(False))
                .alias("Interaction_Outcome_OpenRate")
            )
            if channel_aware
            else (
                pl.when(pl.col.Outcome.is_in(self.ih.positive_outcome_labels["OpenRate"]).any())
                .then(pl.lit(True))
                .when(pl.col.Outcome.is_in(self.ih.negative_outcome_labels["OpenRate"]).any())
                .then(pl.lit(False))
                .alias("Interaction_Outcome_OpenRate")
            )
        )

        other_agg = [
            pl.when(pl.col.Outcome.is_in(self.ih.positive_outcome_labels[metric]).any())
            .then(pl.lit(True))
            .when(pl.col.Outcome.is_in(self.ih.negative_outcome_labels[metric]).any())
            .then(pl.lit(False))
            .alias(f"Interaction_Outcome_{metric}")
            for metric in self.ih.positive_outcome_labels.keys()
            if metric not in ("Engagement", "OpenRate")
        ]

        interactions = (
            cdh_utils._apply_query(source, query)
            .group_by(
                (group_by_clause + ["InteractionID"]) if group_by_clause is not None else ["InteractionID"],
            )
            .agg(
                [engagement_agg, openrate_agg] + other_agg,
                Propensity=pl.col.Propensity.last(),
                # for debugging
                Outcomes=pl.col.Outcome.unique().sort(),
            )
            .drop([] if debug else ["Outcomes"])
        )
        return interactions

    def summary_success_rates(
        self,
        by: str | list[str] | pl.Expr | None = None,
        every: str | timedelta | None = None,
        query: QUERY | None = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Calculate success rates with standard errors.

        Aggregates interactions into success rates for each configured metric,
        with standard errors for statistical significance assessment.

        Parameters
        ----------
        by : str, list[str], or pl.Expr, optional
            Grouping dimension(s). Default is None.
        every : str or timedelta, optional
            Time aggregation period (e.g., "1d", "1w"). Default is None.
        query : QUERY, optional
            Polars expression to filter data before aggregation.
        debug : bool, default False
            If True, include the Outcomes column (list of unique outcome values per group).
            If False, the Outcomes column is dropped from the results.

            This parameter affects the return value structure, not logging output.
            For debug logging, use logging.basicConfig(level=logging.DEBUG).

        Returns
        -------
        pl.LazyFrame
            Success rate summary with columns:

            - **Positives_<metric>**: Count of positive outcomes
            - **Negatives_<metric>**: Count of negative outcomes
            - **Interactions**: Total interaction count
            - **SuccessRate_<metric>**: Positive / (Positive + Negative)
            - **StdErr_<metric>**: Standard error of the proportion
            - Plus any grouping columns

        Notes
        -----
        Standard error is calculated as:

        .. math::

            SE = \\sqrt{\\frac{p(1-p)}{n}}

        where p is the success rate and n is the sample size.

        See Also
        --------
        summarize_by_interaction : Interaction-level outcomes.
        summary_outcomes : Outcome counts.

        Examples
        --------
        >>> ih.aggregates.summary_success_rates(by="Channel", every="1w").collect()

        """
        if not isinstance(by, list):
            by = [by] if by is not None else []  # type: ignore[list-item]
        by_pl_exprs: list[str] = [x.meta.output_name() for x in by if isinstance(x, pl.Expr)]  # type: ignore[attr-defined]
        by_non_pl_exprs = [x for x in by if not isinstance(x, pl.Expr)]
        group_by_clause = cdh_utils.safe_flatten_list(
            by_pl_exprs + by_non_pl_exprs + (["OutcomeTime"] if every is not None else []),
        )

        summary = (
            self.summarize_by_interaction(by, every, query, debug=True)
            .group_by(group_by_clause)
            .agg(
                [
                    pl.col(f"Interaction_Outcome_{metric}")
                    .filter(pl.col(f"Interaction_Outcome_{metric}"))
                    .len()
                    .alias(f"Positives_{metric}")
                    for metric in self.ih.positive_outcome_labels.keys()
                ]
                + [
                    pl.col(f"Interaction_Outcome_{metric}")
                    .filter(pl.col(f"Interaction_Outcome_{metric}").not_())
                    .len()
                    .alias(f"Negatives_{metric}")
                    for metric in self.ih.positive_outcome_labels.keys()
                ],
                Interactions=pl.len(),
                # for debugging
                Outcomes=pl.col.Outcomes.list.explode().unique().sort().drop_nulls(),
            )
            .with_columns(
                [
                    (
                        pl.col(f"Positives_{metric}") / (pl.col(f"Positives_{metric}") + pl.col(f"Negatives_{metric}"))
                    ).alias(f"SuccessRate_{metric}")
                    for metric in self.ih.positive_outcome_labels.keys()
                ],
            )
            .with_columns(
                [
                    (
                        (pl.col(f"SuccessRate_{metric}") * (1 - pl.col(f"SuccessRate_{metric}")))
                        / (pl.col(f"Positives_{metric}") + pl.col(f"Negatives_{metric}"))
                    )
                    .sqrt()
                    .alias(f"StdErr_{metric}")
                    for metric in self.ih.positive_outcome_labels.keys()
                ],
            )
            .drop([] if debug else ["Outcomes"])
        )

        if group_by_clause is None:
            summary = summary.drop("literal")  # created by empty group_by
        else:
            summary = summary.sort(group_by_clause)

        return summary

    def summary_outcomes(
        self,
        by: str | list[str] | pl.Expr | None = None,
        every: str | timedelta | None = None,
        query: QUERY | None = None,
    ) -> pl.LazyFrame:
        """Count outcomes by type.

        Aggregates outcome counts, useful for understanding the distribution
        of outcomes across dimensions or time periods.

        Parameters
        ----------
        by : str, list[str], or pl.Expr, optional
            Grouping dimension(s). Default is None.
        every : str or timedelta, optional
            Time aggregation period (e.g., "1d", "1w"). Default is None.
        query : QUERY, optional
            Polars expression to filter data before aggregation.

        Returns
        -------
        pl.LazyFrame
            Outcome counts with columns:

            - **Outcome**: Outcome type label
            - **Count**: Number of occurrences
            - Plus any grouping columns

        See Also
        --------
        summary_success_rates : Success rates by metric.

        Examples
        --------
        >>> ih.aggregates.summary_outcomes(by="Channel").collect()
        >>> ih.aggregates.summary_outcomes(every="1mo").collect()

        """
        if every is not None:
            source = self.ih.data.with_columns(pl.col.OutcomeTime.dt.truncate(every))
        else:
            source = self.ih.data

        if not isinstance(by, list):
            by = [by] if by is not None else []  # type: ignore[list-item]
        by_pl_exprs: list[str | pl.Expr] = [x for x in by if isinstance(x, pl.Expr)]
        by_non_pl_exprs = [x for x in by if not isinstance(x, pl.Expr)]
        group_by_clause = cdh_utils.safe_flatten_list(
            ["Outcome"] + by_non_pl_exprs + (["OutcomeTime"] if every is not None else []),
            by_pl_exprs,
        )

        summary = (cdh_utils._apply_query(source, query).group_by(group_by_clause).agg(Count=pl.len())).sort("Count")

        return summary

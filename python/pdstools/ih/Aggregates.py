"""Aggregation methods for Interaction History analysis."""

from datetime import timedelta
from typing import TYPE_CHECKING, List, Optional, Union

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
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
        by: Optional[Union[str, List[str], pl.Expr]] = None,
        every: Optional[Union[str, timedelta]] = None,
        query: Optional[QUERY] = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Summarize outcomes per interaction.

        Groups data by interaction ID and determines the outcome for each
        interaction based on configured positive/negative outcome labels.

        Parameters
        ----------
        by : str, List[str], or pl.Expr, optional
            Grouping dimension(s). Default is None.
        every : str or timedelta, optional
            Time aggregation period (e.g., "1d", "1w"). Default is None.
        query : QUERY, optional
            Polars expression to filter data before aggregation.
        debug : bool, default False
            If True, include debug columns (Outcomes list).

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
            by = [by]
        by_pl_exprs = [x for x in by if isinstance(x, pl.Expr)]
        by_non_pl_exprs = [x for x in by if not isinstance(x, pl.Expr)]
        group_by_clause = cdh_utils.safe_flatten_list(
            by_non_pl_exprs + (["OutcomeTime"] if every is not None else []),
            by_pl_exprs,
        )

        interactions = (
            cdh_utils._apply_query(source, query)
            .group_by(
                (group_by_clause + ["InteractionID"])
                if group_by_clause is not None
                else ["InteractionID"]
            )
            .agg(
                # Take only one outcome per interaction. TODO should perhaps be the last one.
                [
                    pl.when(
                        pl.col.Outcome.is_in(
                            self.ih.positive_outcome_labels[metric]
                        ).any()
                    )
                    .then(pl.lit(True))
                    .when(
                        pl.col.Outcome.is_in(
                            self.ih.negative_outcome_labels[metric]
                        ).any()
                    )
                    .then(pl.lit(False))
                    .alias(f"Interaction_Outcome_{metric}")
                    for metric in self.ih.positive_outcome_labels.keys()
                ],
                Propensity=pl.col.Propensity.last(),
                # for debugging
                Outcomes=pl.col.Outcome.unique().sort(),
            )
            .drop([] if debug else ["Outcomes"])
        )
        return interactions

    def summary_success_rates(
        self,
        by: Optional[Union[str, List[str], pl.Expr]] = None,
        every: Optional[Union[str, timedelta]] = None,
        query: Optional[QUERY] = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Calculate success rates with standard errors.

        Aggregates interactions into success rates for each configured metric,
        with standard errors for statistical significance assessment.

        Parameters
        ----------
        by : str, List[str], or pl.Expr, optional
            Grouping dimension(s). Default is None.
        every : str or timedelta, optional
            Time aggregation period (e.g., "1d", "1w"). Default is None.
        query : QUERY, optional
            Polars expression to filter data before aggregation.
        debug : bool, default False
            If True, include debug columns.

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
            by = [by]
        by_pl_exprs = [x.meta.output_name() for x in by if isinstance(x, pl.Expr)]
        by_non_pl_exprs = [x for x in by if not isinstance(x, pl.Expr)]
        group_by_clause = cdh_utils.safe_flatten_list(
            by_pl_exprs
            + by_non_pl_exprs
            + (["OutcomeTime"] if every is not None else [])
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
                        pl.col(f"Positives_{metric}")
                        / (
                            pl.col(f"Positives_{metric}")
                            + pl.col(f"Negatives_{metric}")
                        )
                    ).alias(f"SuccessRate_{metric}")
                    for metric in self.ih.positive_outcome_labels.keys()
                ]
            )
            .with_columns(
                [
                    (
                        (
                            pl.col(f"SuccessRate_{metric}")
                            * (1 - pl.col(f"SuccessRate_{metric}"))
                        )
                        / (
                            pl.col(f"Positives_{metric}")
                            + pl.col(f"Negatives_{metric}")
                        )
                    )
                    .sqrt()
                    .alias(f"StdErr_{metric}")
                    for metric in self.ih.positive_outcome_labels.keys()
                ]
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
        by: Optional[Union[str, List[str], pl.Expr]] = None,
        every: Optional[Union[str, timedelta]] = None,
        query: Optional[QUERY] = None,
    ) -> pl.LazyFrame:
        """Count outcomes by type.

        Aggregates outcome counts, useful for understanding the distribution
        of outcomes across dimensions or time periods.

        Parameters
        ----------
        by : str, List[str], or pl.Expr, optional
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
            by = [by]
        by_pl_exprs = [x for x in by if isinstance(x, pl.Expr)]
        by_non_pl_exprs = [x for x in by if not isinstance(x, pl.Expr)]
        group_by_clause = cdh_utils.safe_flatten_list(
            ["Outcome"]
            + by_non_pl_exprs
            + (["OutcomeTime"] if every is not None else []),
            by_pl_exprs,
        )

        summary = (
            cdh_utils._apply_query(source, query)
            .group_by(group_by_clause)
            .agg(Count=pl.len())
        ).sort("Count")

        return summary

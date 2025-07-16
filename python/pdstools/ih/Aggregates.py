from datetime import timedelta
from typing import TYPE_CHECKING, List, Optional, Union
import polars as pl

from ..utils.namespaces import LazyNamespace
from ..utils import cdh_utils
from ..utils.types import QUERY

if TYPE_CHECKING:
    from .IH import IH as IH_Class


class Aggregates(LazyNamespace):

    def __init__(self, ih: "IH_Class"):
        super().__init__()
        self.ih = ih

    def summarize_by_interaction(
        self,
        by: Optional[Union[str, List[str], pl.Expr]] = None,
        every: Optional[Union[str, timedelta]] = None,
        query: Optional[QUERY] = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Groups the IH data by interaction ID and summarizes outcomes for each interaction.

        It optionally groups by one or more dimensions (e.g. Experiment, Channel, Issue etc). When
        given, the 'every' argument is used to divide the timerange into buckets. It uses the same string
        language as Polars.

        For each interaction, it determines whether any outcomes match the positive or negative outcome labels
        for each metric. An interaction is considered to have a positive outcome for a metric if any of its
        outcomes are in the positive labels for that metric, and negative if any are in the negative labels.

        Parameters
        ----------
        by : Optional[Union[str, List[str], pl.Expr]], optional
            Grouping keys. Name of field(s) or a Polars expression, by default None
        every : Optional[Union[str, timedelta]], optional
            Every interval start and period length, by default None
        query : Optional[QUERY], optional
            Query to filter the data before aggregation, by default None
        debug : bool, optional
            Whether to include debug information in the output, by default False

        Returns
        -------
        pl.LazyFrame
            A polars frame with interaction-level outcome data, including columns for each metric's outcome
            and the propensity value.
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
            by_pl_exprs
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
        """Groups the IH data summarizing into success rates (SuccessRate) and standard error (StdErr).

        It optionally groups by one or more dimensions (e.g. Experiment, Channel, Issue etc). When
        given, the 'every' argument is used to divide the timerange into buckets. It uses the same string
        language as Polars.

        Every interaction is considered to have only one outcome: positive, negative or none. When any
        outcome in the interaction is in the positive labels, the outcome is considered positive. Next,
        when any is in the negative labels, the outcome of the interaction is considered negative. Otherwise
        there is no defined outcome and the interaction is ignored in calculations of success rate or error.

        Parameters
        ----------
        by : Optional[Union[str, List[str], pl.Expr]], optional
            Grouping keys. Name of field(s) or a Polars expression, by default None
        every : Optional[str], optional
            Every interval start and period length, by default None

        Returns
        -------
        pl.LazyFrame
            A polars frame with the grouping keys and columns for the total number of Positives, Negatives,
            number of Interactions, success rate (SuccessRate) and standard error (StdErr).
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
        """Groups the IH data by outcome and summarizes counts for each outcome type.

        It optionally groups by one or more dimensions (e.g. Experiment, Channel, Issue etc). When
        given, the 'every' argument is used to divide the timerange into buckets. It uses the same string
        language as Polars.

        This method provides a count of each outcome type, which can be useful for understanding the
        distribution of outcomes across different dimensions or time periods.

        Parameters
        ----------
        by : Optional[Union[str, List[str], pl.Expr]], optional
            Grouping keys. Name of field(s) or a Polars expression, by default None
        every : Optional[Union[str, timedelta]], optional
            Every interval start and period length, by default None
        query : Optional[QUERY], optional
            Query to filter the data before aggregation, by default None

        Returns
        -------
        pl.LazyFrame
            A polars frame with the grouping keys, outcome types, and a count column showing the number
            of occurrences for each outcome type within each group.
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
            ["Outcome"] + by_non_pl_exprs + (["OutcomeTime"] if every is not None else []),
            by_pl_exprs
        )

        summary = (
            cdh_utils._apply_query(source, query)
            .group_by(group_by_clause)
            .agg(Count=pl.len())
        ).sort("Count")

        return summary

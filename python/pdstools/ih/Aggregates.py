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

    def _summary_interactions(
        self,
        by: Optional[Union[str, List[str]]] = None,
        every: Optional[Union[str, timedelta]] = None,
        query: Optional[QUERY] = None,
    ) -> pl.LazyFrame:
        if every is not None:
            source = self.ih.data.with_columns(pl.col.OutcomeTime.dt.truncate(every))
        else:
            source = self.ih.data

        group_by_clause = cdh_utils.safe_flatten_list(
            [by] + (["OutcomeTime"] if every is not None else [])
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
                Outcomes=pl.col.Outcome.unique().sort(),  # for debugging
            )
        )
        return interactions
    
    def summary_success_rates(
        self,
        by: Optional[Union[str, List[str]]] = None,
        every: Optional[Union[str, timedelta]] = None,
        query: Optional[QUERY] = None,
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
        by : Optional[Union[str, List[str]]], optional
            Grouping keys, by default None
        every : Optional[str], optional
            Every interval start and period length, by default None

        Returns
        -------
        pl.LazyFrame
            A polars frame with the grouping keys and columns for the total number of Positives, Negatives,
            number of Interactions, success rate (SuccessRate) and standard error (StdErr).
        """

        group_by_clause = cdh_utils.safe_flatten_list(
            [by] + (["OutcomeTime"] if every is not None else [])
        )

        summary = (
            self._summary_interactions(by, every, query)
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
                Outcomes=pl.col.Outcomes.list.explode()
                .unique()
                .sort()
                .drop_nulls(),  # for debugging
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
        )

        if group_by_clause is None:
            summary = summary.drop("literal")  # created by empty group_by
        else:
            summary = summary.sort(group_by_clause)

        return summary

    def summary_outcomes(
        self,
        by: Optional[Union[str, List[str]]] = None,
        every: Optional[Union[str, timedelta]] = None,
        query: Optional[QUERY] = None,
    ):

        if every is not None:
            source = self.ih.data.with_columns(pl.col.OutcomeTime.dt.truncate(every))
        else:
            source = self.ih.data

        group_by_clause = cdh_utils.safe_flatten_list(
            ["Outcome"] + [by] + (["OutcomeTime"] if every is not None else [])
        )

        summary = (
            cdh_utils._apply_query(source, query)
            .group_by(group_by_clause)
            .agg(Count=pl.len())
        ).sort(cdh_utils.safe_flatten_list(["Count"]+group_by_clause))

        return summary

from typing import TYPE_CHECKING, List, Optional
import polars as pl

from ..utils.namespaces import LazyNamespace
from ..utils.cdh_utils import safe_flatten_list

if TYPE_CHECKING:
    from .IH import IH as IH_Class


class Aggregates(LazyNamespace):
    def __init__(self, ih: "IH_Class"):
        super().__init__()
        self.ih = ih

    def summary_by_experiment(
        self,
        experiment_field: Optional[str] = None,
        every: Optional[str] = None,
        by: Optional[List[str]] = None,
        positive_labels: Optional[List[str]] = None,
        negative_labels: Optional[List[str]] = None,
    ) -> pl.LazyFrame:
        """Groups the IH data summarizing into success rate (CTR) and standard error (StdErr).

        It groups by the "experiment field" (TODO in the future this can be optional or multiple). When
        given, the 'every' argument is used to divide the timerange into buckets. It uses the same string
        language as Polars.

        Every interaction is considered to have only one outcome: positive, negative or none. When any
        outcome in the interaction is in the positive labels, the outcome is considered positive. Next,
        when any is in the negative labels, the outcome of the interaction is considered negative. Otherwise
        there is no defined outcome and the interaction is ignored in calculations of success rate or error.

        Parameters
        ----------
        experiment_field : Optional[str], optional
            Optional field that contains the experiments
        every : Optional[str], optional
            Every interval start and period length, by default None
        by : Optional[List[str]], optional
            Extra grouping keys, by default None
        positive_labels : Optional[List[str]], optional
            Outcome label(s) for the positive responses, by default None
        negative_labels : Optional[List[str]], optional
            Outcome label(s) for the negative responses, by default None

        Returns
        -------
        pl.LazyFrame
            A polars frame with the grouping keys and columns for the total number of Positives, Negatives,
            number of Interactions, success rate (CTR) and standard error (StdErr).
        """

        if positive_labels is None:
            positive_labels = ["Accepted", "Accept", "Clicked", "Click"]

        if negative_labels is None:
            negative_labels = [
                "Impression",
                "Impressed",
                "Pending",
                "NoResponse",
            ]

        if every is not None:
            source = self.ih.data.with_columns(pl.col.OutcomeTime.dt.truncate(every))
        else:
            source = self.ih.data

        group_by_clause = safe_flatten_list(
            [experiment_field] + [by] + (["OutcomeTime"] if every is not None else [])
        )
        if len(group_by_clause) == 0:
            group_by_clause = None

        summary = (
            source.filter(
                pl.col.ExperimentGroup.is_not_null() & (pl.col.ExperimentGroup != "")
            )
            .group_by(
                (group_by_clause + ["InteractionID"])
                if group_by_clause is not None
                else ["InteractionID"]
            )
            .agg(
                # Take only one outcome per interaction. TODO should perhaps be the last one.
                InteractionOutcome=pl.when(pl.col.Outcome.is_in(positive_labels).any())
                .then(pl.lit(True))
                .when(pl.col.Outcome.is_in(negative_labels).any())
                .then(pl.lit(False)),
                Outcomes=pl.col.Outcome.unique().sort(),  # for debugging
            )
            .group_by(group_by_clause)
            .agg(
                Positives=pl.col.InteractionOutcome.filter(
                    pl.col.InteractionOutcome
                ).len(),
                Negatives=pl.col.InteractionOutcome.filter(
                    pl.col.InteractionOutcome.not_()
                ).len(),
                Interactions=pl.len(),
                Outcomes=pl.col.Outcomes.list.explode()
                .unique()
                .sort()
                .drop_nulls(),  # for debugging
            )
            .with_columns(CTR=pl.col.Positives / (pl.col.Positives + pl.col.Negatives))
            .with_columns(
                StdErr=(
                    (
                        (pl.col.CTR * (1 - pl.col.CTR))
                        / (pl.col.Positives + pl.col.Negatives)
                    ).sqrt()
                )
            )
        )

        if group_by_clause is None:
            summary = summary.drop("literal")  # created by empty group_by
        else:
            summary = summary.sort(group_by_clause)

        return summary

from itertools import chain
from typing import TYPE_CHECKING, Dict, List, Optional
import polars as pl

from ..utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from .IH import IH as IH_Class


class Aggregates(LazyNamespace):
    def __init__(self, ih: "IH_Class"):
        super().__init__()
        self.ih = ih

    def summary_by_experiment(
        self,
        experiment_field: str,
        by: Optional[List[str]] = None,
        positive_labels: List[str] = None,
        negative_labels: List[str] = None,
    ):

        if by is not None:
            if isinstance(by, str):
                by = [by]
        else:
            by = []

        if positive_labels is None:
            positive_labels = ["Accepted", "Accept", "Clicked", "Click"]

        if negative_labels is None:
            negative_labels = [
                "Impression",
                "Impressed",
                "Pending",
                "NoResponse",
            ]

        summary = (
            self.ih.data.filter(
                pl.col.ExperimentGroup.is_not_null() & (pl.col.ExperimentGroup != "")
            )
            .group_by([experiment_field] + by + ["InteractionID"])
            .agg(
                # Take only one outcome per interaction. TODO should perhaps be the last one.
                InteractionOutcome=pl.when(pl.col.Outcome.is_in(positive_labels).any())
                .then(pl.lit(True))
                .when(pl.col.Outcome.is_in(negative_labels).any())
                .then(pl.lit(False)),
                Outcomes=pl.col.Outcome.unique().sort(),  # for debugging
            )
            .group_by([experiment_field] + by)
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
            .sort([experiment_field] + by)
        )

        return summary

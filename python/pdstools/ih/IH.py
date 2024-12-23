import datetime
import os
import random
from typing import Dict, List, Optional, Union
import polars as pl


from .Aggregates import Aggregates
from .Plots import Plots
from ..utils.cdh_utils import _polars_capitalize, _apply_query
from ..utils.types import QUERY
from ..pega_io.File import read_ds_export


class IH:
    data: pl.LazyFrame
    positive_outcome_labels: Dict[str, List[str]]

    def __init__(self, data: pl.LazyFrame):
        self.data = _polars_capitalize(data)

        self.aggregates = Aggregates(ih=self)
        self.plot = Plots(ih=self)
        self.positive_outcome_labels = {
            "Engagement": ["Accepted", "Accept", "Clicked", "Click"],
            "Conversion": ["Conversion"],
        }
        self.negative_outcome_labels = {
            "Engagement": [
                "Impression",
                "Impressed",
                "Pending",
                "NoResponse",
            ],
            "Conversion": ["Impression", "Pending"],
        }

    @classmethod
    def from_ds_export(
        cls,
        ih_filename: Union[os.PathLike, str],
        query: Optional[QUERY] = None,
    ):
        """Import from a Pega Dataset Export"""

        data = read_ds_export(ih_filename).with_columns(
            # TODO this should come from some polars func in utils
            pl.col("pxOutcomeTime").str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z")
        )
        if query is not None:
            data = _apply_query(data, query=query)

        return IH(data)

    @classmethod
    def from_mock_data(cls, days=90, n=100000):
        """Generate sample data"""
        accept_rate = 0.2
        accept_avg_duration_minutes = 10
        convert_over_accept_rate_test = 0.5
        convert_over_accept_rate_control = 0.3
        convert_avg_duration_days = 2

        now = datetime.datetime.now()

        # TODO maybe this should be changed in PDS tools - w/o __TimeStamp__ flag
        # def to_prpc_time_str(__TimeStamp__):
        #     return to_prpc_date_time(__TimeStamp__)[0:15]

        ih_fake_impressions = pl.DataFrame(
            {
                "pxInteractionID": [str(int(1e9 + i)) for i in range(n)],
                "pyChannel": random.choices(["Web", "Email"], k=n),
                "pyIssue": "Acquisition",
                "pyGroup": "Phones",
                "pyName": "AppleIPhone1564GB",
                "ExperimentGroup": ["Conversion-Test", "Conversion-Control"]
                * int(n / 2),
                "pxOutcomeTime": [
                    (now - datetime.timedelta(days=i * days / n)) for i in range(n)
                ],
                "__AcceptDurationMinutes__": [
                    random.uniform(0, 2 * accept_avg_duration_minutes) for i in range(n)
                ],
                "__ConvertDurationDays__": [
                    random.uniform(0, 2 * convert_avg_duration_days) for i in range(n)
                ],
            }
        ).with_columns(
            pyOutcome=pl.when(pl.col.pyChannel == "Web")
            .then(pl.lit("Impression"))
            .otherwise(pl.lit("Pending"))
        )
        ih_fake_accepts = ih_fake_impressions.sample(fraction=accept_rate).with_columns(
            pl.col.pxOutcomeTime
            + pl.duration(minutes=pl.col("__AcceptDurationMinutes__")),
            pyOutcome=pl.when(pl.col.pyChannel == "Web")
            .then(pl.lit("Clicked"))
            .otherwise(pl.lit("Accepted")),
        )
        ih_fake_converts_test = (
            ih_fake_accepts.filter(pl.col.ExperimentGroup == "Conversion-Test")
            .sample(fraction=convert_over_accept_rate_test)
            .with_columns(
                pl.col.pxOutcomeTime
                + pl.duration(days=pl.col("__ConvertDurationDays__")),
                pyOutcome=pl.lit("Conversion"),
            )
        )
        ih_fake_converts_control = (
            ih_fake_accepts.filter(pl.col.ExperimentGroup == "Conversion-Control")
            .sample(fraction=convert_over_accept_rate_control)
            .with_columns(
                pl.col.pxOutcomeTime
                + pl.duration(days=pl.col("__ConvertDurationDays__")),
                pyOutcome=pl.lit("Conversion"),
            )
        )

        ih_data = (
            pl.concat(
                [
                    ih_fake_impressions,
                    ih_fake_accepts,
                    ih_fake_converts_test,
                    ih_fake_converts_control,
                ]
            )
            .filter(pl.col("pxOutcomeTime") < pl.lit(now))
            .drop(
                [
                    "__AcceptDurationMinutes__",
                    "__ConvertDurationDays__",
                ]
            )
            .sort("pxInteractionID", "pxOutcomeTime")
        )

        return IH(ih_data.lazy())

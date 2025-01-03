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
            "OpenRate": ["Opened", "Open"],
        }
        self.negative_outcome_labels = {
            "Engagement": [
                "Impression",
                "Impressed",
                "Pending",
                "NoResponse",
            ],
            "Conversion": ["Impression", "Pending"],
            "OpenRate": ["Impression", "Pending"],
        }

    @classmethod
    def from_ds_export(
        cls,
        ih_filename: Union[os.PathLike, str],
        query: Optional[QUERY] = None,
    ):
        """Create an IH instance from a file with Pega Dataset Export

        Parameters
        ----------
        ih_filename : Union[os.PathLike, str]
            The full path to the dataset files
        query : Optional[QUERY], optional
            An optional argument to filter out selected data, by default None

        Returns
        -------
        IH
            The properly initialized IH object

        """
        data = read_ds_export(ih_filename).with_columns(
            # TODO this should come from some polars func in utils
            pl.col("pxOutcomeTime").str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z")
        )
        if query is not None:
            data = _apply_query(data, query=query)

        return IH(data)

    @classmethod
    def from_s3(cls):
        """Not implemented yet. Please let us know if you would like this functionality!"""
        ...

    @classmethod
    def from_mock_data(cls, days=90, n=100000):
        """Initialize an IH instance with sample data

        Parameters
        ----------
        days : number of days, defaults to 90 days
        n : number of interaction data records, defaults to 100k

        Returns
        -------
        IH
            The properly initialized IH object
        """
        click_rate = 0.2
        accept_rate = 0.15
        click_avg_duration_minutes = 2
        accept_avg_duration_minutes = 30
        convert_over_accept_click_rate_test = 0.5
        convert_over_accept_click_rate_control = 0.3
        convert_avg_duration_days = 2

        now = datetime.datetime.now()

        # TODO maybe this should be changed in PDS tools - w/o __TimeStamp__ flag
        # def to_prpc_time_str(__TimeStamp__):
        #     return to_prpc_date_time(__TimeStamp__)[0:15]

        ih_fake_impressions = pl.DataFrame(
            {
                "pxInteractionID": [str(int(1e9 + i)) for i in range(n)],
                "pyDirection": "",  # will be set later from channel
                "pyChannel": random.choices(["Web", "Email"], k=n),
                "pyIssue": random.choices(
                    ["Acquisition", "Retention", "Risk", "Service"], k=n
                ),
                "pyGroup": random.choices(
                    [
                        "Pension",
                        "Lending",
                        "Mortgages",
                        "Investments",
                        "Insurance",
                        "Savings",
                    ],
                    k=n,
                ),
                "pyName": [
                    random.randint(1, 10) for _ in range(n)
                ],  # nr 1-10 will be appended to group
                "pyTreatment": [
                    random.randint(1, 2) for _ in range(n)
                ],  # nr will be appended to group/channel
                # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
                "__PropensityInbound__": [random.betavariate(1, 10) for _ in range(n)],
                "__PropensityOutbound__": [random.betavariate(1, 20) for _ in range(n)],
                "ExperimentGroup": ["Conversion-Test", "Conversion-Control"]
                * int(n / 2),
                "pxOutcomeTime": [
                    (now - datetime.timedelta(days=i * days / n)) for i in range(n)
                ],
                "__ClickDurationMinutes__": [
                    random.uniform(0, 2 * click_avg_duration_minutes) for i in range(n)
                ],
                "__AcceptDurationMinutes__": [
                    random.uniform(0, 2 * accept_avg_duration_minutes) for i in range(n)
                ],
                "__ConvertDurationDays__": [
                    random.uniform(0, 2 * convert_avg_duration_days) for i in range(n)
                ],
            }
        ).with_columns(
            pyDirection=pl.when(pl.col("pyChannel") == "Web")
            .then(pl.lit("Inbound"))
            .otherwise(pl.lit("Outbound")),
            pyName=pl.format("{}_{}", pl.col("pyGroup"), pl.col("pyName")),
            pyTreatment=pl.format(
                "{}_{}_{}Treatment{}",
                pl.col("pyGroup"),
                pl.col("pyName"),
                pl.col("pyChannel"),
                pl.col("pyTreatment"),
            ),
            pyOutcome=pl.when(pl.col.pyChannel == "Web")
            .then(pl.lit("Impression"))
            .otherwise(pl.lit("Pending")),
            pyPropensity=pl.when(pl.col("pyChannel") == "Web")
            .then(pl.col("__PropensityInbound__")
            * pl.col("pyName").hash()
            / pl.datatypes.UInt64.max())
            .otherwise(pl.col("__PropensityOutbound__")
            * pl.col("pyName").hash()
            / pl.datatypes.UInt64.max()),
        )
        ih_fake_clicks = (
            ih_fake_impressions.filter(pl.col.pyDirection == "Inbound")
            .sample(fraction=click_rate)
            .with_columns(
                pl.col.pxOutcomeTime
                + pl.duration(minutes=pl.col("__ClickDurationMinutes__")),
                pyOutcome=pl.lit("Clicked"),
            )
        )
        ih_fake_accepts = (
            ih_fake_impressions.filter(pl.col.pyDirection == "Outbound")
            .sample(fraction=accept_rate)
            .with_columns(
                pl.col.pxOutcomeTime
                + pl.duration(minutes=pl.col("__AcceptDurationMinutes__")),
                pyOutcome=pl.lit("Accepted"),
            )
        )
        ih_fake_converts_over_clicks_test = (
            ih_fake_clicks.filter(pl.col.ExperimentGroup == "Conversion-Test")
            .sample(fraction=convert_over_accept_click_rate_test)
            .with_columns(
                pl.col.pxOutcomeTime
                + pl.duration(days=pl.col("__ConvertDurationDays__")),
                pyOutcome=pl.lit("Conversion"),
            )
        )
        ih_fake_converts_over_accepts_test = (
            ih_fake_accepts.filter(pl.col.ExperimentGroup == "Conversion-Test")
            .sample(fraction=convert_over_accept_click_rate_test)
            .with_columns(
                pl.col.pxOutcomeTime
                + pl.duration(days=pl.col("__ConvertDurationDays__")),
                pyOutcome=pl.lit("Conversion"),
            )
        )
        ih_fake_converts_over_clicks_control = (
            ih_fake_clicks.filter(pl.col.ExperimentGroup == "Conversion-Control")
            .sample(fraction=convert_over_accept_click_rate_control)
            .with_columns(
                pl.col.pxOutcomeTime
                + pl.duration(days=pl.col("__ConvertDurationDays__")),
                pyOutcome=pl.lit("Conversion"),
            )
        )
        ih_fake_converts_over_accepts_control = (
            ih_fake_accepts.filter(pl.col.ExperimentGroup == "Conversion-Control")
            .sample(fraction=convert_over_accept_click_rate_control)
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
                    ih_fake_clicks,
                    ih_fake_accepts,
                    ih_fake_converts_over_clicks_test,
                    ih_fake_converts_over_clicks_control,
                    ih_fake_converts_over_accepts_test,
                    ih_fake_converts_over_accepts_control,
                ]
            )
            .filter(pl.col("pxOutcomeTime") <= pl.lit(now))
            .drop(
                [
                    "__PropensityInbound__",
                    "__PropensityOutbound__",
                    "__ClickDurationMinutes__",
                    "__AcceptDurationMinutes__",
                    "__ConvertDurationDays__",
                ]
            )
            .sort("pxInteractionID", "pxOutcomeTime")
        )

        return IH(ih_data.lazy())

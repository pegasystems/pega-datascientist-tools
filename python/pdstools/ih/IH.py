import datetime
import os
import random
from typing import Dict, List, Optional, Union
import polars as pl
import polars.selectors as cs

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
        n = int(n)

        n_actions = 10
        click_avg_duration_minutes = 2
        accept_avg_duration_minutes = 30
        convert_over_accept_click_rate_test = 0.5
        convert_over_accept_click_rate_control = 0.3
        convert_avg_duration_days = 2
        inbound_base_propensity = 0.02
        outbound_base_propensity = 0.01
        inbound_modelnoise_NaiveBayes = (
            0.2  # relative amount of extra noise added to models
        )
        inbound_modelnoise_GradientBoost = 0.0
        outbound_modelnoise_NaiveBayes = 0.3
        outbound_modelnoise_GradientBoost = 0.1

        now = datetime.datetime.now()

        def thompson_sampler(propensity, responses=10000):
            # Using the same approach as we're doing in NBAD at the moment
            a = responses * propensity
            b = responses * (1 - propensity)

            sampled = random.betavariate(a, b)

            # print(
            #     f"{a} {b} {propensity} {sampled} {abs((sampled-propensity)/propensity)}"
            # )

            return sampled

        # TODO maybe this should be changed in PDS tools - w/o __TimeStamp__ flag
        # def to_prpc_time_str(__TimeStamp__):
        #     return to_prpc_date_time(__TimeStamp__)[0:15]

        ih_fake_impressions = pl.DataFrame(
            {
                "pxInteractionID": [str(int(1e9 + i)) for i in range(n)],
                "pyChannel": random.choices(
                    ["Web", "Email"], k=n
                ),  # Direction will be derived from this later
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
                    weights=[1, 1, 2, 2, 3, 3],
                    k=n,
                ),
                "pyName": random.choices(
                    range(1, 1 + n_actions),
                    weights=reversed(range(1, 1 + n_actions)),
                    k=n,
                ),  # nr will be appended to group name to form action name
                "pyTreatment": [
                    random.randint(1, 2) for _ in range(n)
                ],  # nr will be appended to group/channel
                # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
                "ExperimentGroup": [
                    "Conversion-Test",
                    "Conversion-Test",
                    "Conversion-Control",
                    "Conversion-Control",
                ]
                * int(n / 4),
                "pyModelTechnique": [
                    "NaiveBayes",
                    "GradientBoost",
                    "NaiveBayes",
                    "GradientBoost",
                ]
                * int(n / 4),
                "pxOutcomeTime": [
                    (now - datetime.timedelta(days=i * days / n)) for i in range(n)
                ],
                "Temp.ClickDurationMinutes": [
                    random.uniform(0, 2 * click_avg_duration_minutes) for i in range(n)
                ],
                "Temp.AcceptDurationMinutes": [
                    random.uniform(0, 2 * accept_avg_duration_minutes) for i in range(n)
                ],
                "Temp.ConvertDurationDays": [
                    random.uniform(0, 2 * convert_avg_duration_days) for i in range(n)
                ],
                "Temp.RandomUniform": [random.uniform(0, 1) for i in range(n)],
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
        )

        action_basepropensities = (
            ih_fake_impressions.group_by(["pyName"])
            .agg(Count=pl.len())
            .sort("Count", descending=True)
            .with_row_index("Index")
            .with_columns(
                (1.0 / (5 + pl.col("Index"))).alias("Temp.Zipf"),
            )
        )

        ih_fake_impressions = (
            ih_fake_impressions.join(
                action_basepropensities.select(["pyName", "Temp.Zipf"]), on=["pyName"]
            )
            .with_columns(
                pl.col("Temp.Zipf")
                .mean()
                .over(["pyChannel", "pyDirection"])
                .alias("Temp.ZipfMean"),
                pl.when(pl.col("pyDirection") == "Inbound")
                .then(pl.lit(inbound_base_propensity))
                .otherwise(pl.lit(outbound_base_propensity))
                .alias("Temp.ChannelBasePropensity"),
            )
            .with_columns(
                BasePropensity=pl.col("Temp.Zipf")
                * pl.col("Temp.ChannelBasePropensity")
                / pl.col("Temp.ZipfMean")
            )
            .with_columns(
                pyPropensity=pl.col("BasePropensity").map_elements(
                    thompson_sampler, pl.Float64
                )
            )
        )

        # Add artificial noise to the models to manipulate some scenarios
        ih_fake_impressions = ih_fake_impressions.with_columns(
            pl.when(
                (pl.col.pyModelTechnique == "NaiveBayes")
                & (pl.col.pyDirection == "Inbound")
            )
            .then(pl.col("Temp.ChannelBasePropensity") * inbound_modelnoise_NaiveBayes)
            .when(
                (pl.col.pyModelTechnique == "GradientBoost")
                & (pl.col.pyDirection == "Inbound")
            )
            .then(
                pl.col("Temp.ChannelBasePropensity") * inbound_modelnoise_GradientBoost
            )
            .when(
                (pl.col.pyModelTechnique == "NaiveBayes")
                & (pl.col.pyDirection == "Outbound")
            )
            .then(pl.col("Temp.ChannelBasePropensity") * outbound_modelnoise_NaiveBayes)
            .when(
                (pl.col.pyModelTechnique == "GradientBoost")
                & (pl.col.pyDirection == "Outbound")
            )
            .then(
                pl.col("Temp.ChannelBasePropensity") * outbound_modelnoise_GradientBoost
            )
            .otherwise(pl.lit(0.0))
            .alias("Temp.ExtraModelNoise")
        )

        ih_fake_clicks = (
            ih_fake_impressions.filter(pl.col.pyDirection == "Inbound")
            .filter(
                pl.col("Temp.RandomUniform")
                < (pl.col("pyPropensity") + pl.col("Temp.ExtraModelNoise"))
            )
            .with_columns(
                pxOutcomeTime=pl.col.pxOutcomeTime
                + pl.duration(minutes=pl.col("Temp.ClickDurationMinutes")),
                pyOutcome=pl.lit("Clicked"),
            )
        )
        ih_fake_accepts = (
            ih_fake_impressions.filter(pl.col.pyDirection == "Outbound")
            .filter(
                pl.col("Temp.RandomUniform")
                < (pl.col("pyPropensity") + pl.col("Temp.ExtraModelNoise"))
            )
            .with_columns(
                pxOutcomeTime=pl.col.pxOutcomeTime
                + pl.duration(minutes=pl.col("Temp.AcceptDurationMinutes")),
                pyOutcome=pl.lit("Accepted"),
            )
        )

        def create_fake_converts(df, group, fraction):
            return (
                df.filter(pl.col("ExperimentGroup") == group)
                .sample(fraction=fraction)
                .with_columns(
                    pxOutcomeTime=pl.col("pxOutcomeTime")
                    + pl.duration(days=pl.col("Temp.ConvertDurationDays")),
                    pyOutcome=pl.lit("Conversion"),
                )
            )

        ih_data = (
            pl.concat(
                [
                    ih_fake_impressions,
                    ih_fake_clicks,
                    ih_fake_accepts,
                    create_fake_converts(
                        ih_fake_clicks,
                        "Conversion-Test",
                        convert_over_accept_click_rate_test,
                    ),
                    create_fake_converts(
                        ih_fake_accepts,
                        "Conversion-Test",
                        convert_over_accept_click_rate_test,
                    ),
                    create_fake_converts(
                        ih_fake_clicks,
                        "Conversion-Control",
                        convert_over_accept_click_rate_control,
                    ),
                    create_fake_converts(
                        ih_fake_accepts,
                        "Conversion-Control",
                        convert_over_accept_click_rate_control,
                    ),
                ]
            )
            .filter(pl.col("pxOutcomeTime") <= pl.lit(now))
            .drop(cs.starts_with("Temp."))
            .sort("pxInteractionID", "pxOutcomeTime")
        )

        return IH(ih_data.lazy())

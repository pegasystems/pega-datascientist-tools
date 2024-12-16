import datetime
import random
import polars as pl
from pdstools.utils import cdh_utils

# Some day will move into a proper IH class

class ih_generator:
    interactions_period_days = 21
    accept_rate = 0.2
    accept_avg_duration_minutes = 10
    convert_over_accept_rate_test = 0.5
    convert_over_accept_rate_control = 0.3
    convert_avg_duration_days = 2

    def generate(self, n):
        now = datetime.datetime.now()


        def _interpolate(min, max, i, n):
            return min + (max - min) * i / (n - 1)


        def to_prpc_time_str(timestamp):
            return cdh_utils.to_prpc_date_time(timestamp)[0:15]


        ih_fake_impressions = pl.DataFrame(
            {
                "InteractionID": [str(int(1e9 + i)) for i in range(n)],
                "TimeStamp": [
                    (now - datetime.timedelta(days=i * self.interactions_period_days / n))
                    for i in range(n)
                ],
                "AcceptDurationMinutes": [
                    random.uniform(0, 2 * self.accept_avg_duration_minutes) for i in range(n)
                ],
                "ConvertDurationDays": [
                    random.uniform(0, 2 * self.convert_avg_duration_days) for i in range(n)
                ],
                "pyChannel": random.choices(["Web"], k=n), # random.choices(["Web", "Email"], k=n),
                "pyIssue": "Acquisition",
                "pyGroup": "Phones",
                "pyName": "AppleIPhone1564GB",
                "ExperimentGroup": ["Conversion-Test", "Conversion-Control"] * int(n / 2),
                "pyOutcome": None,
            }
        ).with_columns(
            pyOutcome=pl.when(pl.col.pyChannel == "Web")
            .then(pl.lit("Impression"))
            .otherwise(pl.lit("Pending"))
        )
        ih_fake_accepts = ih_fake_impressions.sample(fraction=self.accept_rate).with_columns(
            pl.col.TimeStamp + pl.duration(minutes=pl.col("AcceptDurationMinutes")),
            pyOutcome=pl.when(pl.col.pyChannel == "Web")
            .then(pl.lit("Clicked"))
            .otherwise(pl.lit("Accepted")),
        )
        ih_fake_converts_test = ih_fake_accepts.filter(pl.col.ExperimentGroup=="Conversion-Test").sample(
            fraction=self.convert_over_accept_rate_test
        ).with_columns(
            pl.col.TimeStamp + pl.duration(days=pl.col("ConvertDurationDays")),
            pyOutcome=pl.lit("Conversion"),
        )
        ih_fake_converts_control = ih_fake_accepts.filter(pl.col.ExperimentGroup=="Conversion-Control").sample(
            fraction=self.convert_over_accept_rate_control
        ).with_columns(
            pl.col.TimeStamp + pl.duration(days=pl.col("ConvertDurationDays")),
            pyOutcome=pl.lit("Conversion"),
        )

        ih_data=pl.concat([ih_fake_impressions, ih_fake_accepts, ih_fake_converts_test, ih_fake_converts_control]).with_columns(
            pxOutcomeTime=pl.col("TimeStamp").map_elements(
                to_prpc_time_str, return_dtype=pl.String
            ),
        ).filter(pl.col("TimeStamp") < pl.lit(now)).drop(
            ["AcceptDurationMinutes", "ConvertDurationDays", "TimeStamp"]
        ).sort(
            "InteractionID", "pxOutcomeTime"
        ).lazy()

        return ih_data
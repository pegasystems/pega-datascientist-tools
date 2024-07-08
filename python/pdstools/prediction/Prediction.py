import json
from pathlib import Path
from typing import List, Optional, Union
import polars as pl

from ..utils import cdh_utils
from ..utils import NBAD


class Prediction:
    predictions: pl.LazyFrame

    # These are pretty strict conditions - many configurations appear not to satisfy these
    # perhaps the Total = Test + Control is no longer met when Impact Analyzer is around
    prediction_validity_expr = (
        (pl.col("Positives") > 0)
        & (pl.col("Positives_Test") > 0)
        & (pl.col("Positives_Control") > 0)
        & (pl.col("Negatives") > 0)
        & (pl.col("Negatives_Test") > 0)
        & (pl.col("Negatives_Control") > 0)
        # & (
        #     pl.col("Positives")
        #     == (pl.col("Positives_Test") + pl.col("Positives_Control"))
        # )
        # & (
        #     pl.col("Negatives")
        #     == (pl.col("Negatives_Test") + pl.col("Negatives_Control"))
        # )
    )

    def getPredictionsChannelMapping(
        self, custom_predictions: Optional[List[NBAD.NBAD_Prediction]] = None
    ) -> pl.DataFrame:
        if not custom_predictions:
            custom_predictions = []
        all_mappings = NBAD.standardNBADPredictions + custom_predictions
        return pl.DataFrame(
            {
                "Prediction": [x.model_name.upper() for x in all_mappings],
                "Channel": [x.channel for x in all_mappings],
                "Direction": [x.direction for x in all_mappings],
                "isStandardNBADPrediction": [x.standard for x in all_mappings],
                "isMultiChannelPrediction": [x.multi_channel for x in all_mappings],
            }
        )

    def __init__(self, df: pl.LazyFrame):
        predictions_raw_data_prepped = (
            df.filter(pl.col.pyModelType == "PREDICTION")
            .with_columns(
                #                 SnapshotTime=cdh_utils.parsePegaDateTimeFormats(
                #     "SnapshotTime"
                # ).dt.date(),
                SnapshotTime=pl.col("pySnapShotTime")
                .map_elements(
                    lambda x: cdh_utils.fromPRPCDateTime(x), return_dtype=pl.Datetime
                )
                .cast(pl.Date),
                Performance=pl.col("pyValue").cast(pl.Float32),
            )
            .rename(
                {
                    "pyPositives": "Positives",
                    "pyNegatives": "Negatives",
                    "pyCount": "ResponseCount",
                }
            )
        )

        # Below looks like a pivot.. but we want to make sure Control, Test and NBA
        # columns are always there...
        # TODO we may want to assert that this results in exactly one record for
        # every combination of model ID and snapshot time.
        counts_control = predictions_raw_data_prepped.filter(
            pl.col.pyDataUsage == "Control"
        ).select(
            ["pyModelId", "SnapshotTime", "Positives", "Negatives", "ResponseCount"]
        )
        counts_test = predictions_raw_data_prepped.filter(
            pl.col.pyDataUsage == "Test"
        ).select(
            ["pyModelId", "SnapshotTime", "Positives", "Negatives", "ResponseCount"]
        )
        counts_NBA = predictions_raw_data_prepped.filter(
            pl.col.pyDataUsage == "NBA"
        ).select(
            ["pyModelId", "SnapshotTime", "Positives", "Negatives", "ResponseCount"]
        )

        self.predictions = (
            # Performance is taken for the records with a filled in "snapshot type".
            # The numbers of positives, negatives may not make sense but are included
            # anyways.
            predictions_raw_data_prepped.filter(pl.col.pySnapshotType == "Daily")
            .select(
                [
                    "pyModelId",
                    "SnapshotTime",
                    "Positives",
                    "Negatives",
                    "ResponseCount",
                    "Performance",
                ]
            )
            .join(counts_test, on=["pyModelId", "SnapshotTime"], suffix="_Test")
            .join(counts_control, on=["pyModelId", "SnapshotTime"], suffix="_Control")
            .join(counts_NBA, on=["pyModelId", "SnapshotTime"], suffix="_NBA", how="left")
            .with_columns(
                Class=pl.col("pyModelId").str.extract(r"(.+)!.+"),
                ModelName=pl.col("pyModelId").str.extract(r".+!(.+)"),
                CTR=pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives")),
                CTR_Test=pl.col("Positives_Test")
                / (pl.col("Positives_Test") + pl.col("Negatives_Test")),
                CTR_Control=pl.col("Positives_Control")
                / (pl.col("Positives_Control") + pl.col("Negatives_Control")),
                CTR_NBA=pl.col("Positives_NBA")
                / (pl.col("Positives_NBA") + pl.col("Negatives_NBA")),
            )
            .with_columns(
                CTR_Lift=(pl.col("CTR_Test") - pl.col("CTR_Control"))
                / pl.col("CTR_Control"),
                isValidPrediction=self.prediction_validity_expr,
            )
            .sort(["pyModelId", "SnapshotTime"])
        )

    @property
    def is_available(self) -> bool:
        return len(self.predictions.head(1).collect()) > 0

    @property
    def is_valid(self) -> bool:
        return (
            self.is_available
            # or even stronger: pos = pos_test + pos_control
            and self.predictions.select(self.prediction_validity_expr.all())
            .collect()
            .item()
        )

    # TODO generalize the group_by

    def summary_by_channel(
        self,
        custom_predictions: Optional[List[NBAD.NBAD_Prediction]] = None,
        by_period: str = None,
    ) -> pl.LazyFrame:
        """Summarize prediction per channel

        Parameters
        ----------
        custom_predictions : Optional[List[NBAD.NBAD_Prediction]], optional
            Optional list with custom prediction name to channel mappings. Defaults to None.
        by_period : str, optional
            Optional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. If provided, creates a new Period column with the truncated date/time. Defaults to None.

        Returns
        -------
        pl.LazyFrame
            Dataframe with prediction summary (validity, numbers in test, control etc.)
        """
        if not custom_predictions:
            custom_predictions = []

        if by_period is not None:
            period_expr = [
                pl.col("SnapshotTime").dt.truncate(by_period).cast(pl.Date).alias("Period")
            ]
        else:
            period_expr = []

        return (
            self.predictions.join(
                self.getPredictionsChannelMapping(custom_predictions).lazy(),
                left_on="ModelName",
                right_on="Prediction",
                how="left",
            )
            .with_columns(
                [
                    pl.when(pl.col("Channel").is_null())
                    .then(pl.lit("Unknown"))
                    .otherwise(pl.col("Channel")).alias("Channel"),
                    pl.when(pl.col("Direction").is_null())
                    .then(pl.lit("Unknown"))
                    .otherwise(pl.col("Direction")).alias("Direction"),
                    pl.when(pl.col("isStandardNBADPrediction").is_null())
                    .then(pl.lit(False))
                    .otherwise(pl.col("isStandardNBADPrediction")).alias("isStandardNBADPrediction"),
                    pl.when(pl.col("isMultiChannelPrediction").is_null())
                    .then(pl.lit(False))
                    .otherwise(pl.col("isMultiChannelPrediction")).alias("isMultiChannelPrediction"),
                ]
                + period_expr
            )
            .group_by(
                [
                    "ModelName",
                    "Channel",
                    "Direction",
                    "isStandardNBADPrediction",
                    "isMultiChannelPrediction",
                ]
                + (["Period"] if by_period is not None else [])
            )
            .agg(
                cdh_utils.weighted_average_polars("CTR_Lift", "ResponseCount").alias(
                    "Lift"
                ),
                cdh_utils.weighted_performance_polars().alias("Performance"),
                pl.col("Positives").sum(),
                pl.col("Negatives").sum(),
                pl.col("ResponseCount").sum(),
                pl.col("Positives_Test").sum(),
                pl.col("Positives_Control").sum(),
                pl.col("Positives_NBA").sum(),
                pl.col("Negatives_Test").sum(),
                pl.col("Negatives_Control").sum(),
                pl.col("Negatives_NBA").sum(),
            )
            .with_columns(
                usesImpactAnalyzer=(pl.col("Positives_NBA") > 0) & (pl.col("Negatives_NBA") > 0),
                ControlPercentage=100.0
                * (pl.col("Positives_Control") + pl.col("Negatives_Control"))
                / (
                    pl.col("Positives_Test")
                    + pl.col("Negatives_Test")
                    + pl.col("Positives_Control")
                    + pl.col("Negatives_Control")
                    + pl.col("Positives_NBA")
                    + pl.col("Negatives_NBA")
                ),
                TestPercentage=100.0
                * (pl.col("Positives_Test") + pl.col("Negatives_Test"))
                / (
                    pl.col("Positives_Test")
                    + pl.col("Negatives_Test")
                    + pl.col("Positives_Control")
                    + pl.col("Negatives_Control")
                    + pl.col("Positives_NBA")
                    + pl.col("Negatives_NBA")
                ),
                CTR=(pl.col("Positives")) / (pl.col("ResponseCount")),
                isValid=self.prediction_validity_expr,
            )
            .sort(["ModelName"] + (["Period"] if by_period is not None else []))
        )

    # TODO rethink use of multi-channel. If the only valid predictions are multi-channel predictions
    # then use those. If there are valid non-multi-channel predictions then only use those.

    def overall_summary(
        self,
        custom_predictions: Optional[List[NBAD.NBAD_Prediction]] = None,
        by_period: str = None,
    ) -> pl.LazyFrame:
        """Overall prediction summary. Only valid prediction data is included.

        Parameters
        ----------
        custom_predictions : Optional[List[NBAD.NBAD_Prediction]], optional
            Optional list with custom prediction name to channel mappings. Defaults to None.
        by_period : str, optional
            Optional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. If provided, creates a new Period column with the truncated date/time. Defaults to None.

        Returns
        -------
        pl.LazyFrame
            Summary across all valid predictions as a dataframe
        """

        channel_summary = self.summary_by_channel(
            custom_predictions=custom_predictions, by_period=by_period
        )

        if channel_summary.select((pl.col("isMultiChannelPrediction").not_() & pl.col("isValid")).any()).collect().item():
            # There are valid non-multi-channel predictions
            validity_filter_expr = pl.col("isMultiChannelPrediction").not_() & pl.col("isValid")
        else:
            validity_filter_expr = pl.col("isValid")

        return (
            channel_summary
            .filter(validity_filter_expr)
            .group_by(["Period"] if by_period is not None else None)
            .agg(
                pl.concat_str(["Channel", "Direction"], separator="/")
                .n_unique()
                .alias("Number of Valid Channels"),
                cdh_utils.weighted_average_polars("Lift", "ResponseCount").alias(
                    "Overall Lift"
                ),
                cdh_utils.weighted_performance_polars().alias("Performance"),
                pl.col("Positives").sum(),
                pl.col("ResponseCount").sum(),
                pl.col("Channel")
                .where((pl.col("Lift") == pl.col("Lift").min()) & (pl.col("Lift") < 0))
                .first()
                .alias("Channel with Minimum Negative Lift"),
                pl.col("Lift")
                .where((pl.col("Lift") == pl.col("Lift").min()) & (pl.col("Lift") < 0))
                .first()
                .alias("Minimum Negative Lift"),
                pl.col("usesImpactAnalyzer"),
                cdh_utils.weighted_average_polars(
                    "ControlPercentage", "ResponseCount"
                ).alias("ControlPercentage"),
                cdh_utils.weighted_average_polars(
                    "TestPercentage", "ResponseCount"
                ).alias("TestPercentage"),
            )
            .drop(["literal"] if by_period is None else []) # created by null group
            .with_columns(CTR=(pl.col("Positives")) / (pl.col("ResponseCount")),
                          usesImpactAnalyzer=pl.col("usesImpactAnalyzer").list.any())
            .sort(["Period"] if by_period is not None else [])
        )
    

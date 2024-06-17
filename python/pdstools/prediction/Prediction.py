from typing import List, Optional
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
            .join(counts_NBA, on=["pyModelId", "SnapshotTime"], suffix="_NBA")
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
                usesImpactAnalyzer=pl.lit(
                    predictions_raw_data_prepped.select(
                        (pl.col("pyDataUsage") == "NBA").any()
                    )
                    .collect()["pyDataUsage"]
                    .item()
                ),
            )
            .with_columns(
                CTR_Lift=(pl.col("CTR_Test") - pl.col("CTR_Control"))
                / pl.col("CTR_Control"),
                isValidPrediction=self.prediction_validity_expr,
            )
            .sort(["pyModelId", "SnapshotTime"])
        )

    @staticmethod
    def from_pdc(df: pl.LazyFrame):
        return Prediction(
            df.filter(pl.col("ModelType").str.starts_with("Prediction"))
            .rename(
                {
                    "SnapshotTime": "pySnapShotTime",
                    "Positives": "pyPositives",
                    "Negatives": "pyNegatives",
                    "ResponseCount": "pyCount",
                }
            )
            .with_columns(
                pyModelId=pl.format("{}!{}", pl.col("ModelClass"), pl.col("ModelName")),
                pyValue=pl.col("Performance").cast(pl.String),
                pyUnscaledPerformance=(
                    pl.col("Performance").cast(pl.Float64) / 100
                ).cast(pl.String),
                pyName=pl.lit("auc"),
                pyDataUsage=pl.col("ModelType").str.extract(r".+_(Test|Control|NBA)"),
                pyModelType=pl.lit("PREDICTION"),
                pysnapshotday=pl.col("pySnapShotTime").str.slice(0, 8),
                pySnapshotType=pl.lit(
                    "Daily"
                ),  # this is a guess - but Daily is also assumed by the Prediction class
            )
            .cast(
                {
                    "pyNegatives": pl.Float64,
                    "pyPositives": pl.Float64,
                    "pyCount": pl.Float64,
                    # "TotalPositives": pl.Float64,
                    # "TotalResponses": pl.Float64,
                    # "Performance": pl.Float64,
                }
            )
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

    # TODO give this an optional argument for grouping by

    def summary_by_channel(
        self,
        custom_predictions: Optional[List[NBAD.NBAD_Prediction]] = None,
        keep_trend_data: bool = False,
    ) -> pl.LazyFrame:
        if not custom_predictions:
            custom_predictions = []

        return (
            self.predictions.join(
                self.getPredictionsChannelMapping(custom_predictions).lazy(),
                left_on="ModelName",
                right_on="Prediction",
                how="left",
            )
            .with_columns(
                Channel=pl.when(pl.col("Channel").is_null())
                .then(pl.lit("Unknown"))
                .otherwise(pl.col("Channel")),
                Direction=pl.when(pl.col("Direction").is_null())
                .then(pl.lit("Unknown"))
                .otherwise(pl.col("Direction")),
                isStandardNBADPrediction=pl.when(
                    pl.col("isStandardNBADPrediction").is_null()
                )
                .then(pl.lit(False))
                .otherwise(pl.col("isStandardNBADPrediction")),
                isMultiChannelPrediction=pl.when(
                    pl.col("isMultiChannelPrediction").is_null()
                )
                .then(pl.lit(False))
                .otherwise(pl.col("isMultiChannelPrediction")),
            )
            .group_by(
                [
                    "ModelName",
                    "Channel",
                    "Direction",
                    "isStandardNBADPrediction",
                    "isMultiChannelPrediction",
                ]
                + (["SnapshotTime"] if keep_trend_data else [])
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
                pl.col("Negatives_Test").sum(),
                pl.col("Negatives_Control").sum(),
                pl.col("usesImpactAnalyzer").any(),
            )
            .with_columns(
                ControlPercentage=100.0
                * (pl.col("Positives_Control") + pl.col("Negatives_Control"))
                / (
                    pl.col("Positives_Test")
                    + pl.col("Negatives_Test")
                    + pl.col("Positives_Control")
                    + pl.col("Negatives_Control")
                ),
                CTR=(pl.col("Positives")) / (pl.col("ResponseCount")),
                isValidPrediction=self.prediction_validity_expr,
            )
            .sort(["ModelName"] + (["SnapshotTime"] if keep_trend_data else []))
        )

    def overall_summary(
        self, custom_predictions: Optional[List[NBAD.NBAD_Prediction]] = None
    ) -> pl.LazyFrame:
        return (
            self.summary_by_channel(custom_predictions)
            .filter(
                pl.col("isMultiChannelPrediction").not_() & pl.col("isValidPrediction")
            )
            .select(
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
                pl.col("usesImpactAnalyzer").any(),
                cdh_utils.weighted_average_polars(
                    "ControlPercentage", "ResponseCount"
                ).alias("ControlPercentage"),
            )
            .with_columns(CTR=(pl.col("Positives")) / (pl.col("ResponseCount")))
        )

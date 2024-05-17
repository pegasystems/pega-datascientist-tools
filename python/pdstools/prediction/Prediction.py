from typing import List
import polars as pl

from ..utils import cdh_utils, NBAD


class Prediction:
    predictions = None

    # These are pretty strict conditions - many configurations appear not to satisfy these
    # perhaps the Total = Test + Control is no longer met when Impact Analyzer is around
    validityExpr = (
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
        self, custom_predictions=List[NBAD.NBAD_prediction]
    ) -> pl.DataFrame:
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
        self.predictions = (
            df.group_by(
                [
                    "Channel",
                    "ModelName",
                ]
            )
            .agg(
                pl.len(),
                Class=pl.col("ModelClass").unique(),
                Positives=pl.col("Positives")
                .filter(pl.col("ModelType") == "Prediction")
                .sum(),  # Test and Control should add up to this but they do not always do that
                Negatives=pl.col("Negatives")
                .filter(pl.col("ModelType") == "Prediction")
                .sum(),
                Positives_Control=pl.col("Positives")
                .filter(pl.col("ModelType") == "Prediction_Control")
                .sum(),
                Negatives_Control=pl.col("Negatives")
                .filter(pl.col("ModelType") == "Prediction_Control")
                .sum(),
                Positives_Test=pl.col("Positives")
                .filter(pl.col("ModelType") == "Prediction_Test")
                .sum(),
                Negatives_Test=pl.col("Negatives")
                .filter(pl.col("ModelType") == "Prediction_Test")
                .sum(),
                ResponseCount=pl.col("ResponseCount")
                .filter(pl.col("ModelType") == "Prediction")
                .sum(),
                Performance=(pl.col("Performance") * pl.col("ResponseCount"))
                .filter(pl.col("ModelType") == "Prediction")
                .sum()
                / (
                    pl.col("ResponseCount")
                    .filter(pl.col("ModelType") == "Prediction")
                    .sum()
                )
                * 100,
            )
            .with_columns(
                Class=pl.col("Class")
                .list.join(", ")
                .str.to_uppercase(),  # Prediction entries are uppercase so lets align
                CTR=pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives")),
                CTR_Test=pl.col("Positives_Test")
                / (pl.col("Positives_Test") + pl.col("Negatives_Test")),
                CTR_Control=pl.col("Positives_Control")
                / (pl.col("Positives_Control") + pl.col("Negatives_Control")),
            )
            .with_columns(
                CTR_Lift=(pl.col("CTR_Test") - pl.col("CTR_Control"))
                / pl.col("CTR_Control"),
                isValidPrediction=self.validityExpr,
            )
            .sort(["ModelName"])
        )

    @property
    def is_available(self) -> bool:
        return len(self.predictions.head(1).collect()) > 0

    @property
    def is_valid(self) -> bool:
        return (
            self.is_available
            # or even stronger: pos = pos_test + pos_control
            and self.predictions.select(self.validityExpr.all()).collect().item()
        )

    def summary_by_channel(
        self,
        custom_predictions: List[NBAD.NBAD_prediction] = None,
    ) -> pl.LazyFrame:
        if not custom_predictions:
            custom_predictions = []

        return (
            self.predictions.drop("Channel")
            .join(
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
                    "isValidPrediction",
                ]
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
            )
            .with_columns(CTR=(pl.col("Positives")) / (pl.col("ResponseCount")))
            .sort(["ModelName"])
        )

    def overall_summary(
        self, custom_predictions: List[NBAD.NBAD_prediction] = None
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
            )
            .with_columns(CTR=(pl.col("Positives")) / (pl.col("ResponseCount")))
        )

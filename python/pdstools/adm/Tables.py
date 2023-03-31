from . import ADMDatamart
import polars as pl
from functools import cached_property

standardNBADNames = [
    "Assisted_Click_Through_Rate",
    "CallCenter_Click_Through_Rate",
    "CallCenterAcceptRateOutbound",
    "Default_Inbound_Model",
    "Default_Outbound_Model",
    "Email_Click_Through_Rate",
    "Mobile_Click_Through_Rate",
    "OmniAdaptiveModel",
    "Other_Inbound_Click_Through_Rate",
    "Push_Click_Through_Rate",
    "Retail_Click_Through_Rate",
    "Retail_Click_Through_Rate_Outbound",
    "SMS_Click_Through_Rate",
    "Web_Click_Through_Rate",
]


class Tables:
    @cached_property
    def _by(self):
        columns = [
            "Configuration",
            "Issue",
            "Group",
            "Name",
            "Channel",
            "Direction",
            "Treatment",
        ]

        return [
            col
            for col in columns
            if col in self.modelData.columns
            and self.modelData.schema[col] != pl.Null
        ]
    @property
    def AvailableTables(self):
        df = pl.DataFrame(
            {
                "model_overview": [1, 0],
                "predictors_per_configuration": [1, 1],
                "bad_predictors": [1, 1],
                "zero_response": [1, 0],
                "zero_positives": [1, 0],
                "reach": [1, 0],
                "minimum_performance": [1, 1],
                "appendix": [1, 0],
            }
        )
        df = df.transpose().with_columns(pl.Series(df.columns))
        df.columns = ["modelData", "predictorData", "Tables"]
        return df.select(["Tables", "modelData", "predictorData"])
    
    @property
    def ApplicableTables(self):
        df = self.AvailableTables
        if not self.hasModels:
            df = df.filter(pl.col("modelData") == 0)
        if not self.hasPredictorBinning:
            df = df.filter(pl.col("predictorData") == 0)
        return df.get_column("Tables").to_list()

    @cached_property
    def model_overview(self):
        return (
            self.last(strategy="lazy")
            .groupby(["Configuration", "Channel", "Direction"])
            .agg(
                [
                    pl.col("Name").unique().count().alias("Number of Actions"),
                    pl.col("ModelID").unique().count().alias("Number of Models"),
                ]
            )
            .with_columns(
                [
                    pl.col("Configuration")
                    .is_in(standardNBADNames)
                    .alias("Standard in NBAD Framework"),
                    (pl.col("Number of Models") / pl.col("Number of Actions"))
                    .round(2)
                    .alias("Average number of Treatments per Action"),
                ]
            )
            .sort("Configuration")
        ).collect()

    @cached_property
    def predictors_per_configuration(self):
        return (
            self.combinedData.groupby("Configuration")
            .agg(
                [
                    pl.col("PredictorName").unique().count().alias("Predictor Count"),
                    pl.col("Channel").unique().alias("Used in (Channels)"),
                    pl.col("Issue").unique().alias("Used for (Issues)"),
                ]
            )
            .collect()
        )

    @cached_property
    def bad_predictors(self):
        return (
            self.predictorData.filter(pl.col("PredictorName") != "Classifier")
            .groupby("PredictorName")
            .agg(
                [
                    pl.sum("ResponseCount").alias("Response Count"),
                    (pl.min("Performance") * 100).alias("Min"),
                    (pl.mean("Performance") * 100).alias("Mean"),
                    (pl.median("Performance") * 100).alias("Median"),
                    (pl.max("Performance") * 100).alias("max"),
                ]
            )
            .sort("Mean", descending=False)
        ).collect()

    @property
    def _zero_response(self):
        return self.modelData.groupby(self._by).agg(
            [pl.sum("ResponseCount"), pl.sum("Positives"), pl.mean("Performance")]
        )

    @cached_property
    def zero_response(self):
        return self._zero_response.filter(pl.col("ResponseCount") == 0).collect()

    @cached_property
    def zero_positives(self):
        return self._zero_response.filter(pl.col("Positives") == 0).collect()

    @cached_property
    def _last_counts(self):
        return (
            self.last(strategy="lazy")
            .groupby(self._by)
            .agg([pl.sum("ResponseCount"), pl.sum("Positives"), pl.mean("Performance")])
        )

    @cached_property
    def reach(self):
        def calc_reach(x=pl.col("Positives")):
            return 0.02 + 0.98 * (pl.min([pl.lit(200), x]) / 200)

        return (
            self._last_counts.filter(
                (pl.col("Positives") < 200) & (pl.col("Positives") > 0)
            )
            .with_columns(Reach=calc_reach())
            .collect()
        )

    @cached_property
    def minimum_performance(self):
        return self._last_counts.filter(
            (pl.col("Positives") >= 200) & (pl.col("Performance") == 0.5)
        ).collect()

    @cached_property
    def appendix(self):
        return (
            self.modelData.groupby(self._by + ["ModelID"])
            .agg(
                [
                    pl.max("ResponseCount").alias("Responses"),
                    pl.count("SnapshotTime").alias("Snapshots"),
                ]
            )
            .collect()
        )

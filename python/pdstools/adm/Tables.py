import polars as pl
from ..utils.cdh_utils import weighted_average_polars
from functools import cached_property

# TODO: reconsier this whole class

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
            if col in self.modelData.columns and self.modelData.schema[col] != pl.Null
        ]

    @property
    def AvailableTables(self):
        df = pl.DataFrame(
            {
                "modeldata_last_snapshot": [1, 0],
                "predictor_last_snapshot": [1, 1], # TODO how is this used?
                "predictorbinning": [1, 1],
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

    @property
    def ApplicableTablesNoPredictorBinning(self):
        applicable_tables = self.ApplicableTables
        if "predictorbinning" in applicable_tables:
            applicable_tables.remove("predictorbinning")
        return applicable_tables

    @cached_property
    def modeldata_last_snapshot(self):
        modeldata_last_snapshot = self.last(table="modelData")

        return modeldata_last_snapshot

    # TODO does this summarization really belong here?
    
    @cached_property
    def predictor_last_snapshot(self):
        model_identifiers = ["Configuration"] + self.context_keys

        predictor_summary = (
            self.last(table="predictorData")
            .filter(pl.col("PredictorName") != "Classifier")
            .join(
                self.last("modelData").select(["ModelID"] + model_identifiers).unique(),
                on="ModelID",
                how="left",
            )
            .group_by(
                model_identifiers + ["ModelID", "PredictorName"]
            )  # Going to model-predictor level removing bins
            .agg(
                pl.first("Type"),
                pl.first("Performance"),
                pl.count("BinIndex").alias("Bins"),
                pl.col("BinResponseCount")
                .where(pl.col("BinType") == "MISSING")
                .sum()
                .alias("Missing"),
                pl.col("BinResponseCount")
                .where(pl.col("BinType") == "RESIDUAL")
                .sum()
                .alias("Residual"),
                pl.first("Positives"),
                pl.first("ResponseCount"),
            )
            .group_by(
                model_identifiers + ["PredictorName"]
            )  # Going to contextkeys-predictor level removing model
            .agg(
                pl.first("Type"),
                weighted_average_polars("Performance", "ResponseCount"),
                weighted_average_polars("Bins", "ResponseCount"),
                ((pl.sum("Missing") / pl.sum("ResponseCount")) * 100).alias(
                    "Missing %"
                ),
                ((pl.sum("Residual") / pl.sum("ResponseCount")) * 100).alias(
                    "Residual %"
                ),
                pl.sum("Positives"),
                pl.sum("ResponseCount").alias("Responses"),
            )
            .fill_null(0)
            .fill_nan(0)
            .with_columns(pl.col("Bins").cast(pl.Int16))
        )

        return predictor_summary

    @cached_property
    def predictorbinning(self):
        return self.last(table="combinedData").filter(
            pl.col("PredictorName") != "Classifier"
        )

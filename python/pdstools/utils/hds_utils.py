import os
import re
import glob
import json


import polars as pl

from typing import Literal
from pdstools import ADMDatamart
from random import randint


class Config:
    def __init__(
        self,
        config_file=None,
        hdr_folder=".",
        use_datamart=False,
        datamart_folder="datamart",
        output_format="ndjson",
        output_folder="output",
        mapping_file="mapping.map",
        mask_predictor_names=True,
        mask_context_key_names=True,
        mask_ih_names=True,
        mask_outcome_name=True,
        mask_predictor_values=True,
        mask_context_key_values=True,
        mask_ih_values=True,
        mask_outcome_values=True,
        context_key_label="Context_*",
        ih_label="IH_*",
        outcome_column="Decision_Outcome",
        positive_outcomes=["Accepted", "Clicked"],
        negative_outcomes=["Rejected", "Impression"],
        special_predictors=[
            "Decision_DecisionTime",
            "Decision_OutcomeTime",
        ],
    ):
        self._opts = {key: value for key, value in vars().items() if key != "self"}
        if config_file is not None:
            self.load_from_config_file(config_file)
        for key, value in self._opts.items():
            setattr(self, key, value)
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

    def load_from_config_file(self, config_file):
        if not os.path.exists(config_file):
            raise ValueError("Config file does not exist.")
        with open(config_file) as f:
            self._opts = json.load(f)

    def save_to_config_file(self):
        with open("config.json", "w") as f:
            json.dump(self._opts, f)


class DataAnonymization:
    def __init__(
        self,
        config=None,
        df: pl.LazyFrame = None,
        datamart: ADMDatamart = None,
        **config_args,
    ):

        self.config = Config(**config_args) if config is None else config

        if df is None:
            self.df = self.load_hdr_files()
        elif isinstance(df, pl.DataFrame):
            self.df = df.lazy()
        elif isinstance(df, pl.LazyFrame):
            self.df = self.df
        else:
            raise ValueError("No data found.")
        self.df_out = None
        if self.config.use_datamart:
            self.predictors_by_type = self.read_predictor_type_from_datamart(
                config.datamart_folder, datamart
            )
        else:
            self.predictors_by_type = self.read_predictor_type_from_file(self.df)

        (
            self.symbolic_predictors_to_mask,
            self.numeric_predictors_to_mask,
            self.column_mapping,
        ) = self.get_predictors_mapping()

    def write_to_output(
        self, df=None, ext: Literal["ndjson", "parquet", "arrow", "csv"] = None
    ):
        out_filename = self.config.output_filename
        out_format = self.config.output_format if ext is None else ext
        if df is None:
            df = self.process()
        if out_format == "ndjson":
            df.write_ndjson(f"{out_filename}.json")
        elif out_format == "parquet":
            df.write_parquet(f"{out_filename}.parquet")
        elif out_format == "arrow":
            df.write_ipc(f"{out_filename}.arrow")
        else:
            df.write_csv(f"{out_filename}.csv")

    def create_mapping_files(self):
        with open(self.config.mapping_filename, "w") as wr:
            for key, mapped_val in self.column_mapping.items():
                wr.write(f"{key}={mapped_val}")
                wr.write("\n")

    def load_hdr_files(self):
        files = glob.glob(f"{self.config.hdr_folder}/*.json")
        if len(files) < 1:
            raise FileNotFoundError(
                """Files not found. Please check the `hdr_folder` 
                and check if there are `.json` files in there. 
                If preferred, you can also supply a polars Dataframe or Lazyframe 
                directly with the `df` argument."""
            )
        out = []
        for file in files:
            out.append(pl.scan_ndjson(file))

        df = pl.concat(out, how="diagonal")
        return df

    @staticmethod
    def read_predictor_type_from_file(df):
        types = {}
        from numpy import random

        count = df.with_row_count().tail(1).select("row_nr").collect().item() + 1
        msk = [bool(i) for i in random.rand(count) < 1 - 0.3][0:50]
        df_ = df.filter(msk).collect()

        for col in df_.columns:
            try:
                df_.get_column(col).cast(pl.Float64)
                types[col] = "numeric"
            except:
                types[col] = "symbolic"

        return types

    @staticmethod
    def read_predictor_type_from_datamart(datamart_folder, datamart=None):
        dm = ADMDatamart(path=datamart_folder) if datamart is None else datamart
        df = pl.DataFrame(dm.predictorData)
        df = (
            df.filter(pl.col("EntryType") != "Classifier")
            .unique(subset="PredictorName")
            .select(["PredictorName", "Type"])
        )
        df_dict = dict(zip(df.get_column("PredictorName"), df.get_column("Type")))
        df_dict = dict(
            (key.replace(".", "_"), value) for (key, value) in df_dict.items()
        )
        return df_dict

    def get_columns_by_type(self):
        columns = self.df.columns
        columns_ = ",".join(columns)

        exp = "[^,]+"
        context_keys_t = re.findall(f"{self.config.context_key_label}{exp}", columns_)
        ih_predictors_t = re.findall(f"{self.config.ih_label}{exp}", columns_)

        special_predictors_t = []
        for exc in self.config.special_predictors:
            for e in re.findall(exc, columns_):
                special_predictors_t.append(e)

        outcome_column_t = [self.config.outcome_column]
        predictors_t = list(
            set(columns)
            - set(
                context_keys_t
                + ih_predictors_t
                + special_predictors_t
                + outcome_column_t
            )
        )
        return context_keys_t, ih_predictors_t, special_predictors_t, predictors_t

    def get_predictors_mapping(self):
        symbolic_predictors_to_mask, numeric_predictors_to_mask = [], []

        (
            context_keys_t,
            ih_predictors_t,
            special_predictors_t,
            predictors_t,
        ) = self.get_columns_by_type()
        outcome_t = self.config.outcome_column

        outcome_column = {}
        context_keys = {}
        ih_predictors = {}
        predictors = {}

        for idx, val in enumerate(context_keys_t):
            if self.config.mask_context_key_values:
                if self.predictors_by_type[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            context_keys[val] = (
                f"CK_PREDICTOR_{idx}" if self.config.mask_context_key_names else val
            )

        for idx, val in enumerate(ih_predictors_t):
            if self.config.mask_ih_values:
                if self.predictors_by_type[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            ih_predictors[val] = (
                f"IH_PREDICTOR_{idx}" if self.config.mask_ih_names else val
            )

        for idx, val in enumerate(predictors_t):
            if self.config.mask_predictor_values:
                if self.predictors_by_type[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            predictors[val] = (
                f"PREDICTOR_{idx}" if self.config.mask_predictor_names else val
            )

        special_predictors = {x: x for x in special_predictors_t}

        outcome_column[outcome_t] = (
            "OUTCOME" if self.config.mask_outcome_name else outcome_t
        )

        column_mapping = (
            predictors
            | context_keys
            | ih_predictors
            | special_predictors
            | outcome_column
        )

        return symbolic_predictors_to_mask, numeric_predictors_to_mask, column_mapping

    def process(self):
        def to_hash(cols):
            return (
                pl.when(
                    (pl.col(cols).is_not_null())
                    & (pl.col(cols) != "")
                    & (pl.col(cols).is_in(["true", "false"]).is_not())
                )
                .then(pl.col(cols).hash(seed=0))
                .otherwise(pl.col(cols))
            )

        def to_normalize(cols):
            return (pl.col(cols) - pl.col(cols).min()) / (
                pl.col(cols).max() - pl.col(cols).min()
            )

        def to_boolean(col, positives):
            return (
                pl.when(pl.col(col).is_in(positives)).then(True).otherwise(False)
            ).alias(col)

        return (
            self.df.with_columns(
                pl.col(self.numeric_predictors_to_mask).cast(pl.Float64)
            )
            .with_columns(
                [
                    to_hash(self.symbolic_predictors_to_mask),
                    to_normalize(self.numeric_predictors_to_mask),
                    to_boolean(
                        self.config.outcome_column, self.config.positive_outcomes
                    ),
                ]
            )
            .select(self.column_mapping.keys())
            .rename(self.column_mapping)
        ).collect()

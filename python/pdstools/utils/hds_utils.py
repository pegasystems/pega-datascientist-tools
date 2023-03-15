import os
import re
import glob
import json
from pathlib import Path
import polars as pl

from random import randint
from typing import Literal, Optional, Union
from pdstools import ADMDatamart


class Config:
    """Configuration file for the data anonymizer.

    Parameters
    ----------
    config_file: str = None
        An optional path to a config file
    hds_folder: Path = "."
        The path to the hds files
    use_datamart: bool = False
        Whether to use the datamart to infer predictor types
    datamart_folder: Path = "datamart"
        The folder of the datamart files
    output_format: Literal["ndjson", "parquet", "arrow", "csv"] = "ndjson"
        The output format to write the files in
    output_folder: Path = "output"
        The location to write the files to
    mapping_file: str = "mapping.map"
        The name of the predictor mapping file
    mask_predictor_names: bool = True
        Whether to mask the names of regular predictors
    mask_context_key_names: bool = True
        Whether to mask the names of context key predictors
    mask_ih_names: bool = True
        Whether to mask the name of Interaction History summary predictors
    mask_outcome_name: bool = True
        Whether to mask the name of the outcome column
    mask_predictor_values: bool = True
        Whether to mask the values of regular predictors
    mask_context_key_values: bool = True
        Whether to mask the values of context key predictors
    mask_ih_values: bool = True
        Whether to mask the values of Interaction History summary predictors
    mask_outcome_values: bool = True
        Whether to mask the values of the outcomes to binary
    context_key_label: str = "Context_*"
        The pattern of names for context key predictors
    ih_label: str = "IH_*"
        The pattern of names for Interaction History summary predictors
    outcome_column: str = "Decision_Outcome"
        The name of the outcome column
    positive_outcomes: list = ["Accepted", "Clicked"]
        Which positive outcomes to map to True
    negative_outcomes: list = ["Rejected", "Impression"]
        Which negative outcomes to map to False
    special_predictors: list = ["Decision_DecisionTime", "Decision_OutcomeTime"]
        A list of special predictors which are not touched
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        hds_folder: Path = ".",
        use_datamart: bool = False,
        datamart_folder: Path = "datamart",
        output_format: Literal["ndjson", "parquet", "arrow", "csv"] = "ndjson",
        output_folder: Path = "output",
        mapping_file: str = "mapping.map",
        mask_predictor_names: bool = True,
        mask_context_key_names: bool = True,
        mask_ih_names: bool = True,
        mask_outcome_name: bool = True,
        mask_predictor_values: bool = True,
        mask_context_key_values: bool = True,
        mask_ih_values: bool = True,
        mask_outcome_values: bool = True,
        context_key_label: str = "Context_*",
        ih_label: str = "IH_*",
        outcome_column: str = "Decision_Outcome",
        positive_outcomes: list = ["Accepted", "Clicked"],
        negative_outcomes: list = ["Rejected", "Impression"],
        special_predictors: list = ["Decision_DecisionTime", "Decision_OutcomeTime"],
    ):
        self._opts = {key: value for key, value in vars().items() if key != "self"}

        if config_file is not None:
            self.load_from_config_file(config_file)
        for key, value in self._opts.items():
            setattr(self, key, value)
        self.validate_paths()

    def load_from_config_file(self, config_file: Path):
        """Load the configurations from a file.

        Parameters
        ----------
        config_file : Path
            The path to the configuration file
        """
        if not os.path.exists(config_file):
            raise ValueError("Config file does not exist.")
        with open(config_file) as f:
            self._opts = json.load(f)

    def save_to_config_file(self, file_name: str = None):
        """Save the configurations to a file.

        Parameters
        ----------
        file_name : str
            The name of the configuration file
        """
        with open("config.json" if file_name is None else file_name, "w") as f:
            json.dump(self._opts, f)

    def validate_paths(self):
        """Validate the outcome folder exists."""
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)


class DataAnonymization:
    """Anonymize a historical dataset.

    Parameters
    ----------
    config
        Override the default configurations with the Config class
    df
        Manually supply a Polars lazyframe to anonymize
    datamart
        Manually supply a Datamart file to infer predictor types

    Keyword arguments
    -----------------
    **config_args
        See :Class:`.Config`

    Example
    -------
    See https://pegasystems.github.io/pega-datascientist-tools/Python/articles/Example_Data_Anonymization.html
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        df: Optional[pl.LazyFrame] = None,
        datamart: Optional[ADMDatamart] = None,
        **config_args,
    ):
        self.config = Config(**config_args) if config is None else config

        if df is None:
            self.df = self.load_hds_files()
        elif isinstance(df, pl.DataFrame):
            self.df = df.lazy()
        elif isinstance(df, pl.LazyFrame):
            self.df = df
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
        self,
        df: Optional[pl.DataFrame] = None,
        ext: Literal["ndjson", "parquet", "arrow", "csv"] = None,
    ):
        """Write the processed dataframe to an output file.

        Parameters
        ----------
        df : Optional[pl.DataFrame]
            Dataframe to write.
            If not provided, runs `self.process()`
        ext : Literal["ndjson", "parquet", "arrow", "csv"]
            What extension to write the file to

        """
        out_filename = f"{self.config.output_folder}/hds"
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

    def create_mapping_file(self):
        """Create a file to write the column mapping"""
        with open(self.config.mapping_file, "w") as wr:
            for key, mapped_val in self.column_mapping.items():
                wr.write(f"{key}={mapped_val}")
                wr.write("\n")

    def load_hds_files(self):
        """Load the historical dataset files from the `config.hds_folder` location."""
        files = glob.glob(f"{self.config.hds_folder}/*.json")
        if len(files) < 1:
            raise FileNotFoundError(
                """Files not found. Please check the `hds_folder` 
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
    def read_predictor_type_from_file(df: pl.LazyFrame):
        """Infer the types of the preditors from the data.

        This is non-trivial, as it's not ideal to pull in all data to memory for this.
        For this reason, we sample 1% of data, or all data if less than 50 rows,
        and try to cast it to numeric. If that fails, we set it to categorical,
        else we set it to numeric.

        It is technically supported to manually override this, by just overriding
        the `symbolic_predictors_to_mask` & `numeric_predictors_to_mask` properties.

        Parameters
        ----------
        df : pl.LazyFrame
            The lazyframe to infer the types with

        """
        types = {}

        import numpy as np

        def sample_it(s: pl.Series) -> pl.Series:
            out = pl.Series(
                values=np.random.binomial(1, 0.01, s.len()),
                dtype=pl.Boolean,
            )
            if out.len() < 50:
                out = pl.Series(values=[True] * out.len(), dtype=pl.Boolean)
            return out

        df_ = (
            df.lazy()
            .with_columns(pl.first().map(sample_it).alias("_sample"))
            .filter(pl.col("_sample"))
            .drop("_sample")
            .collect()
        )

        for col in df_.columns:
            try:
                df_.get_column(col).cast(pl.Float64)
                types[col] = "numeric"
            except:
                types[col] = "symbolic"

        return types

    @staticmethod
    def read_predictor_type_from_datamart(
        datamart_folder: Path, datamart: ADMDatamart = None
    ):
        """The datamart contains type information about each predictor.
        This function extracts that information to infer types for the HDS.

        Parameters
        ----------
        datamart_folder : Path
            The path to the datamart files
        datamart : ADMDatamart
            The direct ADMDatamart object
        """
        dm = ADMDatamart(path=datamart_folder) if datamart is None else datamart
        df = dm.predictorData
        df = (
            df.filter(pl.col("EntryType") != "Classifier")
            .unique(subset="PredictorName")
            .select(["PredictorName", "Type"])
        ).collect()
        df_dict = dict(zip(df.get_column("PredictorName"), df.get_column("Type")))
        df_dict = dict(
            (key.replace(".", "_"), value) for (key, value) in df_dict.items()
        )
        return df_dict

    def get_columns_by_type(self):
        """Get a list of columns for each type."""
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
        """Map the predictor names to their anonymized form."""
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

        column_mapping = {
            **predictors,
            **context_keys,
            **ih_predictors,
            **special_predictors,
            **outcome_column,
        }

        return symbolic_predictors_to_mask, numeric_predictors_to_mask, column_mapping

    def getHasher(
        self,
        cols,
        algorithm="xxhash",
        seed: Union[Literal[0, "random"], int] = 0,
        seed_1=None,
        seed_2=None,
        seed_3=None,
    ):
        if algorithm == "xxhash":
            if seed == "random":
                seed, seed_1, seed_2, seed_3 = (
                    randint(0, 1000000000),
                    randint(0, 1000000000),
                    randint(0, 1000000000),
                    randint(0, 1000000000),
                )

            return pl.col(cols).hash(
                seed=seed, seed_1=seed_1, seed_2=seed_2, seed_3=seed_3
            )

        else:
            return pl.col(cols).apply(algorithm)

    def process(self, **kwargs):
        """Anonymize the dataset."""

        def to_hash(cols, **kwargs):
            hasher = self.getHasher(cols, **kwargs)
            return (
                pl.when(
                    (pl.col(cols).is_not_null())
                    & (pl.col(cols) != "")
                    & (pl.col(cols).is_in(["true", "false"]).is_not())
                )
                .then(hasher)
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

        df = self.df.with_columns(
            pl.col(self.numeric_predictors_to_mask).cast(pl.Float64)
        ).with_columns(
            [
                to_hash(
                    self.symbolic_predictors_to_mask,
                    **kwargs,
                ),
                to_normalize(self.numeric_predictors_to_mask),
            ]
        )
        if self.config.mask_outcome_values:
            df = df.with_columns(
                to_boolean(self.config.outcome_column, self.config.positive_outcomes)
            )
        df = (
            df.select(self.column_mapping.keys()).rename(self.column_mapping)
        ).collect()
        return df

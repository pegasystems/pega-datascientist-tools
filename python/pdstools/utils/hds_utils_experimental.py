from pathlib import Path
import polars as pl
import numpy as np
import glob
import os
import re
from typing import Literal, Optional
from itertools import chain
import json
from tqdm.auto import tqdm


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
    sample_percentage_schema_inferencing: float
        The percentage of records to sample to infer the column type.
        In case you're getting casting errors, it may be useful to
        increase this percentage to check a larger portion of data.
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
        mask_predictor_names: bool = False,
        mask_context_key_names: bool = False,
        mask_ih_names: bool = True,
        mask_outcome_name: bool = False,
        mask_predictor_values: bool = True,
        mask_context_key_values: bool = False,
        mask_ih_values: bool = True,
        mask_outcome_values: bool = True,
        context_key_label: str = "Context_*",
        ih_label: str = "IH_*",
        outcome_column: str = "Decision_Outcome",
        positive_outcomes: list = ["Accepted", "Clicked"],
        negative_outcomes: list = ["Rejected", "Impression"],
        special_predictors: list = [
            "Decision_DecisionTime",
            "Decision_OutcomeTime",
            "Decision_Rank",
        ],
        sample_percentage_schema_inferencing: float = 0.01,
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


class DataAnonymization2:
    def __init__(self, files, config=None, **config_args):
        self.config = Config(**config_args) if config is None else config

        if isinstance(files, str) or isinstance(files, Path):
            self.df = self.getFiles(files)
            cols = [df.columns for df in self.df.values()]
            self.allColumns = set(chain(*cols))
        else:
            self.df = files.lazy()
            self.allColumns = set(files.columns)

        self.typeMapping = None
        self.numericRanges = None

    def inferTypes(self, number_of_files_to_sample=20):
        if isinstance(self.df, pl.LazyFrame):
            self._inferTypes(self.df)
        elif isinstance(self.df, dict):
            if number_of_files_to_sample > len(self.df):
                number_of_files_to_sample = len(self.df)

            filesToCheck = np.round(
                np.linspace(0, len(self.df) - 1, number_of_files_to_sample)
            ).astype(int)

            filesToCheck = [list(self.df.values())[i] for i in filesToCheck]

            for column in self.allColumns:
                files_with_column = [
                    f for f in list(self.df.values()) if column in f.columns
                ]
                number_of_sampled_files_with_column = len(
                    [f for f in filesToCheck if column in f.columns]
                )
                if number_of_sampled_files_with_column < 5:
                    if len(files_with_column) > 5:
                        sampledFiles = np.round(
                            np.linspace(0, len(files_with_column) - 1, 5)
                        ).astype(int)
                        filesToCheck += [
                            list(self.df.values())[i] for i in sampledFiles
                        ]
                    else:
                        filesToCheck += files_with_column

            df = pl.concat(filesToCheck, how="diagonal").collect()
            self.typeMapping = self._inferTypes(df)

    def _inferTypes(self, df):
        types = dict()
        for col in df.columns:
            try:
                ser = df.get_column(col)
                try:
                    ser = ser.map_dict({"": None}, default=pl.first())
                except Exception as e:
                    pass
                ser.cast(pl.Float64)
                types[col] = "numeric"
            except Exception as e:
                types[col] = "symbolic"

        return types

    def setPredictorTypes(self, typeDict: dict):
        self.typeMapping = typeDict

    def setTypeForPredictor(self, predictor, type: Literal["numeric", "symbolic"]):
        self.typeMapping[predictor] = type

    def getPredictorsWithTypes(self):
        return self.typeMapping

    def getFiles(self, orig_files):
        """Load the historical dataset files from the `config.hds_folder` location."""
        files = glob.glob(f"{orig_files}/*.json")
        files = [f for f in files if os.path.getsize(f) > 0]

        if len(files) < 1:
            raise FileNotFoundError(
                """Files not found. Please check the `hds_folder` 
                and check if there are `.json` files in there. 
                If preferred, you can also supply a polars Dataframe or Lazyframe 
                directly with the `df` argument."""
            )
        return {f: pl.scan_ndjson(f) for f in files}

    def create_mapping_file(self):
        """Create a file to write the column mapping"""
        if self.column_mapping is None:
            self.create_predictor_mapping()
        with open(self.config.mapping_file, "w") as wr:
            for key, mapped_val in self.column_mapping.items():
                wr.write(f"{key}={mapped_val}")
                wr.write("\n")

    def get_columns_by_type(self):
        """Get a list of columns for each type."""
        columns = self.allColumns
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

    def create_predictor_mapping(self):
        """Map the predictor names to their anonymized form."""
        symbolic_predictors_to_mask, numeric_predictors_to_mask = [], []

        (
            context_keys_t,
            ih_predictors_t,
            special_predictors_t,
            predictors_t,
        ) = self.get_columns_by_type()

        if not hasattr(self, "typeMapping"):
            self.inferTypes()

        outcome_t = self.config.outcome_column

        outcome_column = {}
        context_keys = {}
        ih_predictors = {}
        predictors = {}

        for idx, val in enumerate(context_keys_t):
            if self.config.mask_context_key_values:
                if self.typeMapping[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            context_keys[val] = (
                f"CK_PREDICTOR_{idx}" if self.config.mask_context_key_names else val
            )

        for idx, val in enumerate(ih_predictors_t):
            if self.config.mask_ih_values:
                if self.typeMapping[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            ih_predictors[val] = (
                f"IH_PREDICTOR_{idx}" if self.config.mask_ih_names else val
            )

        for idx, val in enumerate(predictors_t):
            if self.config.mask_predictor_values:
                if self.typeMapping[val] == "symbolic":
                    symbolic_predictors_to_mask.append(val)
                else:
                    numeric_predictors_to_mask.append(val)
            predictors[val] = (
                f"PREDICTOR_{idx}" if self.config.mask_predictor_names else val
            )

        special_predictors = {x: x for x in special_predictors_t}
        if "Decision_SubjectID" in special_predictors_t:
            symbolic_predictors_to_mask.append("Decision_SubjectID")

        outcome_column[outcome_t] = (
            "OUTCOME" if self.config.mask_outcome_name else outcome_t
        )

        self.column_mapping = {
            **predictors,
            **context_keys,
            **ih_predictors,
            **special_predictors,
            **outcome_column,
        }
        self.symbolic_predictors_to_mask = symbolic_predictors_to_mask
        self.numeric_predictors_to_mask = numeric_predictors_to_mask

    def calculateNumericRanges(self, verbose=False):
        def minMaxFrame(df, columns_to_check):
            col = (
                pl.col(columns_to_check)
                .map_dict({"": None}, default=pl.first())
                .cast(pl.Float64)
            )
            return df.select(
                col.min().suffix("_min"),
                col.max().suffix("_max"),
            ).collect()

        if not hasattr(self, "numeric_predictors_to_mask"):
            self.create_predictor_mapping()

        if len(self.numeric_predictors_to_mask) == 0:
            self.ranges = {}
            return None

        if hasattr(self, "ranges"):
            if all(self.numeric_predictors_to_mask in set(self.ranges.keys())):
                return None
            else:
                columns_to_check = list(
                    col
                    for col in self.numeric_predictors_to_mask
                    if col not in self.ranges.keys()
                )
        else:
            columns_to_check = self.numeric_predictors_to_mask

        if isinstance(self.df, pl.LazyFrame):
            try:
                rangeFrame = minMaxFrame(self.df, columns_to_check)
            except Exception as e:
                if verbose:
                    print("Error when building MinMaxFrame: ", e)
        else:
            rangeList = []
            for file in tqdm(
                self.df.values(), desc="Calculating numeric min/max ranges"
            ):
                try:
                    rangeList.append(minMaxFrame(file, columns_to_check))
                except Exception as e:
                    print(f"Exception {e} for file {file}.")
            if verbose:
                print("Calculated all min/max ranges")
            rangeFrame = pl.concat(rangeList, how="diagonal")
            if verbose:
                print("Combined them into a single frame")

        out = rangeFrame.select(
            pl.col("^*min$").cast(pl.Float64).min(),
            pl.col("^*max$").cast(pl.Float64).max(),
        ).to_dict(as_series=False)
        # print("Got global min/max: ", out)
        minmax = {}
        for col in self.numeric_predictors_to_mask:
            try:
                minmax[col] = {
                    "min": float(out[f"{col}_min"][0]),
                    "max": float(out[f"{col}_max"][0]),
                }
            except (ValueError, TypeError) as e:
                if verbose:
                    print(
                        f'Unable to parse column {col}. Min value found is {out[f"{col}_min"]}, max value {out[f"{col}_max"]}. Setting to categorical. Error: {e}'
                    )
                self.setTypeForPredictor(col, "symbolic")
                self.numeric_predictors_to_mask.remove(col)
                self.symbolic_predictors_to_mask.append(col)

        self.ranges = minmax

    def setNumericRanges(self, rangeDict: dict):
        if not hasattr(self, "ranges"):
            self.ranges = {}
        for key, value in rangeDict.items():
            self.ranges[key] = value

    def setRangeForPredictor(self, predictorName, min, max):
        if not hasattr(self, "ranges"):
            self.ranges = {}
        self.ranges[predictorName] = {"min": min, "max": max}

    def write_to_output(
        self,
        df: Optional[pl.DataFrame] = None,
        ext: Literal["ndjson", "parquet", "arrow", "csv"] = None,
        mode: Literal["optimized", "robust"] = "optimized",
        on_failed_file: Literal["warn", "ignore", "fail"] = "fail",
        verbose=False,
    ):
        """Write the processed dataframe to an output file.

        Parameters
        ----------
        df : Optional[pl.DataFrame]
            Dataframe to write.
            If not provided, runs `self.process()`
        ext : Literal["ndjson", "parquet", "arrow", "csv"]
            What extension to write the file to
        mode : Literal['optimized', 'robust'], default = 'optimized'
            Whether to output a single file (optimized) or maintain
            the same file structure as the original files (robust).
            Optimized should be faster, but robust should allow for bigger
            data as we don't need all data in memory at the same time.

        """
        out_filename = f"{self.config.output_folder}/hds"
        out_format = self.config.output_format if ext is None else ext

        if df is None:
            df = self.process(strategy="lazy")
        if mode == "robust":
            if isinstance(self.df, pl.LazyFrame):
                mode = "optimized"
            else:
                if not hasattr(self, "ranges"):
                    self.calculateNumericRanges(verbose=verbose)
                for filename, df in tqdm(self.df.items(), desc="Anonymizing..."):
                    newName = Path(out_filename) / Path(filename).name
                    if not os.path.exists(newName.parent):
                        os.mkdir(newName.parent)
                    init_df = self.process(df, verbose=verbose)
                    if init_df is None:
                        if on_failed_file == "warn":
                            print(f"Failed to parse {filename}. Continuing")
                        if on_failed_file == "fail":
                            raise Exception(
                                f'Could not parse {filename}. Set on_failed_file to "warn" to continue'
                            )
                    else:
                        init_df.write_ndjson(newName)

        if mode == "optimized":
            df = df.collect()
            if out_format == "ndjson":
                df.write_ndjson(f"{out_filename}.json")
            elif out_format == "parquet":
                df.write_parquet(f"{out_filename}.parquet")
            elif out_format == "arrow":
                df.write_ipc(f"{out_filename}.arrow")
            else:
                df.write_csv(f"{out_filename}.csv")

    def getHasher(
        self,
        cols,
        algorithm="xxhash",
        seed="random",
        seed_1=None,
        seed_2=None,
        seed_3=None,
    ):
        if algorithm == "xxhash":
            if seed == "random":
                if not hasattr(self, "seeds"):
                    from random import randint

                    self.seeds = dict(
                        seed=randint(0, 1000000000),
                        seed_1=randint(0, 1000000000),
                        seed_2=randint(0, 1000000000),
                        seed_3=randint(0, 1000000000),
                    )

            return pl.col(cols).hash(**self.seeds)

        else:
            return pl.col(cols).apply(algorithm)

    def to_normalize(self, cols, verbose=False):
        self.normalizationFailures, exprs = [], []
        for col in cols:
            try:
                exprs.append(
                    (pl.col(col) - pl.lit(float(self.ranges[col]["min"])))
                    / pl.lit(
                        float(self.ranges[col]["max"]) - float(self.ranges[col]["min"])
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"Error when trying to normalize {col}: {e}")
                self.normalizationFailures.append(col)
        return exprs

    def process(
        self,
        df=None,
        strategy="eager",
        verbose=False,
        **kwargs,
    ):
        """Anonymize the dataset."""
        if self.typeMapping is None:
            self.inferTypes()
        if not hasattr(self, "column_mapping"):
            self.create_predictor_mapping()

        if df is None:
            if isinstance(self.df, dict):
                df = pl.concat(self.df.values())
            else:
                df = self.df

        column_mapping = {
            key: val for key, val in self.column_mapping.items() if key in df.columns
        }

        if not hasattr(self, "ranges"):
            self.calculateNumericRanges(verbose=verbose)

        def to_hash(cols, **kwargs):
            hasher = self.getHasher(cols, **kwargs)
            return (
                pl.when(
                    (pl.col(cols).is_not_null())
                    & (pl.col(cols).cast(pl.Utf8) != pl.lit(""))
                    & (pl.col(cols).cast(pl.Utf8).is_in(["true", "false"]).is_not())
                )
                .then(hasher)
                .otherwise(pl.col(cols))
            )

        def to_boolean(col, positives):
            return (
                pl.when(pl.col(col).cast(pl.Utf8).is_in(positives))
                .then(True)
                .otherwise(False)
            ).alias(col)

        numerics_in_file = [
            col for col in self.numeric_predictors_to_mask if col in df.columns
        ]
        symbolics_in_file = [
            col for col in self.symbolic_predictors_to_mask if col in df.columns
        ]

        try:
            df = df.with_columns(
                pl.col(numerics_in_file)
                .map_dict({"": None}, default=pl.first())
                .cast(pl.Float64)
            ).with_columns(
                [
                    to_hash(
                        symbolics_in_file,
                        **kwargs,
                    ),
                    *self.to_normalize(numerics_in_file, verbose=verbose),
                ]
            )
            if self.config.mask_outcome_values:
                df = df.with_columns(
                    to_boolean(
                        self.config.outcome_column, self.config.positive_outcomes
                    )
                )
            if len(self.normalizationFailures) > 0 and verbose:
                print(
                    f"{len(self.normalizationFailures)} could not be normalized: {self.normalizationFailures}"
                )
            column_mapping = {
                key: val
                for key, val in column_mapping.items()
                if key not in self.normalizationFailures
            }
            df = df.select(column_mapping.keys()).rename(column_mapping)

            if strategy == "eager":
                return df.lazy().collect()
            return df
        except Exception as e:
            if verbose:
                print(f"Error when anonymizing: {e}. Available columns: {df.columns}.")
            return None

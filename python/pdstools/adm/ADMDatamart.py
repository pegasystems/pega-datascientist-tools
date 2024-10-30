from __future__ import annotations

import datetime
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union

import polars as pl

from .. import pega_io
from ..pega_io.File import read_dataflow_output, read_ds_export
from ..utils import cdh_utils
from ..utils.cdh_utils import _polars_capitalize
from ..utils.types import QUERY
from . import Schema
from .ADMTrees import AGB
from .Aggregates import Aggregates
from .BinAggregator import BinAggregator
from .CDH_Guidelines import CDHGuidelines
from .Plots import Plots
from .Reports import Reports

logger = logging.getLogger(__name__)


class ADMDatamart:
    """
    The main class for interacting with ADM data from the Pega Datamart.

    To initialize this class, either
    1. Initialize directly with the model_df and predictor_df polars LazyFrames
    2. Use one of the class methods: `from_ds_export`, `from_s3` or `from_dataflow_export`

    This class will read in the data from different sources, properly structure them
    from further analysis, and apply correct typing and useful renaming.

    There is also a few "namespaces" that you can call from this class:
    - `.plot` contains ready-made plots to analyze the data with
    - `.aggregates` contains mostly internal data aggregations queries
    - `.agb` contains analysis utilities for Adaptive Gradient Boosting models
    - `.generate` leads to some ready-made reports, such as the Health Check
    - `.bin_aggregator` allows you to compare the bins across various models

    Parameters
    ----------
    model_df : pl.LazyFrame, optional
        The Polars LazyFrame representation of the model snapshot table.
    predictor_df : pl.LazyFrame, optional
        The Polars LazyFrame represenation of the predictor binning table.
    query : QUERY, optional
        An optional query to apply to the input data.
        For details, see :meth:`pdstools.cdh_utils._apply_query`.
    extract_pyname_keys : bool, default = True
        Whether to extract extra keys from the `pyName` column.

    See Also
    --------
    pdstools.adm.Plots : The out of the box plots to run.
    pdstools.adm.Reports : The Health Check and Model Report files to generate.
    pdstools.cdh_utils._apply_query : The internal query mechansm to filter the data.

    Examples
    --------
    >>> from pdstools import ADMDatamart
    >>> from glob import glob
    >>> dm = ADMDatamart(
             model_df = pl.scan_parquet('models.parquet'),
             predictor_df = pl.scan_parquet('predictors.parquet')
             query = {"Configuration":["Web_Click_Through"]}
             )
    >>> dm = ADMDatamart.from_ds_export(base_path='/my_export_folder')
    >>> dm = ADMDatamart.from_s3("pega_export")
    >>> dm = ADMDatamart.from_dataflow_export(glob("data/models*"), glob("data/preds*"))
    """

    model_data: Optional[pl.LazyFrame]
    predictor_data: Optional[pl.LazyFrame]
    combined_data: Optional[pl.LazyFrame]

    def __init__(
        self,
        model_df: Optional[pl.LazyFrame] = None,
        predictor_df: Optional[pl.LazyFrame] = None,
        *,
        query: Optional[QUERY] = None,
        extract_pyname_keys: bool = True,
    ) -> None:
        self.context_keys: List[str] = [
            "Channel",
            "Direction",
            "Issue",
            "Group",
            "Name",
        ]

        self.plot = Plots(datamart=self)
        self.aggregates = Aggregates(datamart=self)
        self.agb = AGB(datamart=self)
        self.generate = Reports(datamart=self)
        self.cdh_guidelines = CDHGuidelines()

        self.model_data = self._validate_model_data(
            model_df, query=query, extract_pyname_keys=extract_pyname_keys
        )

        # TODO shouldnt we subset the predictor data to the model IDs also in the model data - if that is present
        self.predictor_data = self._validate_predictor_data(predictor_df)

        self.combined_data = self.aggregates._combine_data(
            self.model_data, self.predictor_data
        )

        self.bin_aggregator = BinAggregator(dm=self)  # attach after model_data

    @classmethod
    def from_ds_export(
        cls,
        model_filename: Optional[str] = None,
        predictor_filename: Optional[str] = None,
        base_path: Union[os.PathLike, str] = ".",
        *,
        query: Optional[QUERY] = None,
        extract_pyname_keys: bool = True,
    ):
        model_df = read_ds_export(model_filename or "model_data", base_path)
        predictor_df = read_ds_export(predictor_filename or "predictor_data", base_path)
        return cls(
            model_df, predictor_df, query=query, extract_pyname_keys=extract_pyname_keys
        )

    @classmethod
    def from_s3(cls): ...

    @classmethod
    def from_dataflow_export(
        cls,
        model_data_files: Union[Iterable[str], str],
        predictor_data_files: Union[Iterable[str], str],
        *,
        query: Optional[QUERY] = None,
        extract_pyname_keys: bool = True,
        cache_file_prefix: str = "",
        extension: Literal["json"] = "json",
        compression: Literal["gzip"] = "gzip",
        cache_directory: Union[os.PathLike, str] = "cache",
    ):
        model_data = read_dataflow_output(
            model_data_files,
            cache_file_prefix + "model_data",
            extension=extension,
            compression=compression,
            cache_directory=cache_directory,
        )

        predictor_data = read_dataflow_output(
            predictor_data_files,
            cache_file_prefix + "predictor_data",
            extension=extension,
            compression=compression,
            cache_directory=cache_directory,
        )
        return cls(
            model_df=model_data,
            predictor_df=predictor_data,
            query=query,
            extract_pyname_keys=extract_pyname_keys,
        )

    def _validate_model_data(
        self,
        df: Optional[pl.LazyFrame],
        query: Optional[QUERY] = None,
        extract_pyname_keys: bool = True,
    ) -> Optional[pl.LazyFrame]:
        if df is None:
            logger.info("No model data available.")
            return df

        df = _polars_capitalize(df)
        schema = df.collect_schema()
        if extract_pyname_keys and "Name" in schema.names():
            df = cdh_utils._extract_keys(df)

        if "Treatment" in schema.names():
            self.context_keys.append("Treatment")

        self.context_keys = [k for k in self.context_keys if k in schema.names()]

        df = df.with_columns(
            SuccessRate=(pl.col("Positives") / pl.col("ResponseCount")).fill_nan(
                pl.lit(0)
            ),
        )
        if not isinstance(schema["SnapshotTime"], pl.Datetime):
            df = df.with_columns(SnapshotTime=cdh_utils.parse_pega_date_time_formats())

        df = cdh_utils._apply_schema_types(df, Schema.ADMModelSnapshot)

        return cdh_utils._apply_query(df, query)

    def _validate_predictor_data(
        self, df: Optional[pl.LazyFrame]
    ) -> Optional[pl.LazyFrame]:
        if df is None:
            logger.info("No predictor data available.")
            return df
        df = _polars_capitalize(df)
        schema = df.collect_schema()

        if "BinResponseCount" not in schema.names():  # pragma: no cover
            df = df.with_columns(
                BinResponseCount=(pl.col("BinPositives") + pl.col("BinNegatives"))
            )
        df = df.with_columns(
            BinPropensity=pl.col("BinPositives") / pl.col("BinResponseCount"),
            BinAdjustedPropensity=(
                (pl.col("BinPositives") + pl.lit(0.5))
                / (pl.col("BinResponseCount") + pl.lit(1))
            ),
        )

        if "PredictorCategory" not in schema.names():
            df = self.apply_predictor_categorization(df)
        df = cdh_utils._apply_schema_types(df, Schema.ADMPredictorBinningSnapshot)
        return df

    def apply_predictor_categorization(
        self,
        df: Optional[pl.LazyFrame] = None,
        categorization: Optional[
            Union[pl.Expr, Callable[..., pl.Expr]]
        ] = cdh_utils.default_predictor_categorization,
    ):
        if callable(categorization):
            categorization: pl.Expr = categorization()

        if df is not None:
            return df.with_columns(PredictorCategory=categorization)

        if hasattr(self, "predictor_data") and self.predictor_data is not None:
            self.predictor_data = self.predictor_data.with_columns(
                PredictorCategory=categorization
            )
        if hasattr(self, "combined_data") and self.combined_data is not None:
            self.combined_data = self.combined_data.with_columns(
                PredictorCategory=categorization
            )

    def save_data(
        self, path: Union[os.PathLike, str] = "."
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Cache modelData and predictorData to files.

        Parameters
        ----------
        path : str
            Where to place the files

        Returns
        -------
        (Optional[Path], Optional[Path]):
            The paths to the model and predictor data files
        """
        abs_path = Path(path).resolve()
        time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]
        modeldata_cache, predictordata_cache = None, None
        if self.model_data is not None:
            modeldata_cache = pega_io.cache_to_file(
                self.model_data, abs_path, name=f"cached_model_data_{time}"
            )
        if self.predictor_data is not None:
            predictordata_cache = pega_io.cache_to_file(
                self.predictor_data, abs_path, name=f"cached_predictor_data_{time}"
            )

        return modeldata_cache, predictordata_cache

    @cached_property
    def unique_channels(self):
        return set(
            self.model_data.select(pl.col("Channel").unique().sort()).collect()[
                "Channel"
            ]
        )

    @cached_property
    def unique_configurations(self):
        return set(
            self.model_data.select(pl.col("Configuration").unique())
            .collect()["Configuration"]
            .to_list()
        )

    @cached_property
    def unique_channel_direction(self):
        return set(
            self.model_data.select(
                pl.concat_str(pl.col("Channel"), pl.col("Direction"), separator="/")
                .unique()
                .alias("ChannelDirection")
            )
            .collect()["ChannelDirection"]
            .to_list()
        )

    @cached_property
    def unique_configuration_channel_direction(self):
        return set(
            self.model_data.select(
                pl.concat_str(
                    pl.col("Configuration"),
                    pl.col("Channel"),
                    pl.col("Direction"),
                    separator="/",
                )
                .unique()
                .alias("ChannelDirection")
            )
            .collect()["ChannelDirection"]
            .to_list()
        )

    @cached_property
    def unique_predictor_categories(self):
        return set(
            self.predictor_data.select(
                pl.col("PredictorCategory").unique().sort()
            ).collect()["PredictorCategory"]
        )

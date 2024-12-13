from __future__ import annotations

__all__ = ["ADMDatamart"]

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
    Monitor and analyze ADM data from the Pega Datamart.

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
        For details, see :meth:`pdstools.utils.cdh_utils._apply_query`.
    extract_pyname_keys : bool, default = True
        Whether to extract extra keys from the `pyName` column.
        In older Pega versions, this contained pyTreatment among other
        (customizable) fields. By default True

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

    Note
    ----
    This class depends on two datasets:

    - `pyModelSnapshots` corresponds to the `model_data` attribute
    - `pyADMPredictorSnapshots` corresponds to the `predictor_data` attribute

    For instructions on how to download these datasets, please refer to the following
    article: https://docs.pega.com/bundle/platform/page/platform/decision-management/exporting-monitoring-database.html

    See Also
    --------
    pdstools.adm.Plots : The out of the box plots on the Datamart data
    pdstools.adm.Reports : Methods to generate the Health Check and Model Report
    pdstools.utils.cdh_utils._apply_query : How to query the ADMDatamart class and methods
    """

    model_data: Optional[pl.LazyFrame]
    predictor_data: Optional[pl.LazyFrame]
    combined_data: Optional[pl.LazyFrame]
    plot: Plots
    aggregates: Aggregates
    agb: AGB
    generate: Reports
    cdh_guidelines: CDHGuidelines
    bin_aggregator: BinAggregator

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
        self.cdh_guidelines = (
            CDHGuidelines()
        )  # not sure if this should be part of the ADM DM

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
        """Import the ADMDatamart class from a Pega Dataset Export

        Parameters
        ----------
        model_filename : Optional[str], optional
            The full path or name (if base_path is given) to the model snapshot files,
            by default None
        predictor_filename : Optional[str], optional
            The full path or name (if base_path is given) to the predictor binning
            snapshot files, by default None
        base_path : Union[os.PathLike, str], optional
            A base path to provide so that we can automatically find the most recent
            files for both the model and predictor snapshots, if model_filename and
            predictor_filename are not given as full paths, by default "."
        query : Optional[QUERY], optional
            An optional argument to filter out selected data, by default None
        extract_pyname_keys : bool, optional
            Whether to extract additional keys from the `pyName` column, by default True

        Returns
        -------
        ADMDatamart
            The properly initialized ADMDatamart class

        Examples
        --------
        >>> from pdstools import ADMDatamart

        >>> # To automatically find the most recent files in the 'my_export_folder' dir:
        >>> dm = ADMDatamart.from_ds_export(base_path='/my_export_folder')

        >>> # To specify individual files:
        >>> dm = ADMDatamart.from_ds_export(
                model_df='/Downloads/model_snapshots.parquet',
                predictor_df = '/Downloads/predictor_snapshots.parquet'
                )

        Note
        ----
        By default, the dataset export in Infinity returns a zip file per table.
        You do not need to open up this zip file! You can simply point to the zip,
        and this method will be able to read in the underlying data.

        See Also
        --------
        pdstools.pega_io.File.read_ds_export : More information on file compatibility
        pdstools.utils.cdh_utils._apply_query : How to query the ADMDatamart class and methods
        """
        # "model_data"/"predictor_data" are magic keywords
        # to automatically find data files in base_path
        model_df = read_ds_export(model_filename or "model_data", base_path)
        predictor_df = read_ds_export(predictor_filename or "predictor_data", base_path)
        return cls(
            model_df, predictor_df, query=query, extract_pyname_keys=extract_pyname_keys
        )

    @classmethod
    def from_s3(cls):
        """Not implemented yet. Please let us know if you would like this functionality!"""
        ...

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
        """Read in data generated by a data flow, such as the Prediction Studio export.

        Dataflows are able to export data from and to various sources.
        As they are meant to be used in production, they are highly resiliant.
        For every partition and every node, a dataflow will output a small json file
        every few seconds. While this is great for production loads, it can be a bit
        more tricky to read in the data for smaller-scale and ad-hoc analyses.

        This method aims to make the ingestion of such highly partitioned data easier.
        It reads in every individual small json file that the dataflow has output,
        and caches them to a parquet file in the `cache_directory` folder.
        As such, if you re-run this method later with more data added since the last
        export, we will not read in from the (slow) dataflow files, but rather from the
        (much faster) cache.

        Parameters
        ----------
        model_data_files : Union[Iterable[str], str]
            A list of files to read in as the model snapshots
        predictor_data_files : Union[Iterable[str], str]
            A list of files to read in as the predictor snapshots
        query : Optional[QUERY], optional
            A, by default None
        extract_pyname_keys : bool, optional
            Whether to extract extra keys from the pyName column, by default True
        cache_file_prefix : str, optional
            An optional prefix for the cache files, by default ""
        extension : Literal[&quot;json&quot;], optional
            The extension of the source data, by default "json"
        compression : Literal[&quot;gzip&quot;], optional
            The compression of the source files, by default "gzip"
        cache_directory : Union[os.PathLike, str], optional
            Where to store the cached files, by default "cache"

        Returns
        -------
        ADMDatamart
            An initialized instance of the datamart class

        Examples
        --------
        >>> from pdstools import ADMDatamart
        >>> import glob
        >>> dm = ADMDatamart.from_dataflow_export(glob("data/models*"), glob("data/preds*"))

        See also
        --------
        pdstools.utils.cdh_utils._apply_query : How to query the ADMDatamart class and methods
        glob : Makes creating lists of files much easier
        """
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
        """Internal method to validate model data"""
        if df is None:
            logger.info("No model data available.")
            return df

        df = _polars_capitalize(df)
        schema = df.collect_schema()
        if extract_pyname_keys and "Name" in schema.names():
            df = cdh_utils._extract_keys(df)

        if "Treatment" in schema.names():
            self.context_keys.append("Treatment")

        # Model technique (NaiveBayes or GradientBoost) added in '24 (US-648869 and related)
        if "ModelTechnique" not in schema.names():
            df = df.with_columns(
                ModelTechnique=pl.lit(None),
            )
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
        """Internal method to validate predictor data"""
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
        categorization: Union[
            pl.Expr, Callable[..., pl.Expr]
        ] = cdh_utils.default_predictor_categorization,
    ):
        """Apply a new predictor categorization to the datamart tables

        In certain plots, we use the predictor categorization to indicate what 'kind'
        a certain predictor is, such as IH, Customer, etc. Call this method with a
        custom Polars Expression (or a method that returns one) - and it will be applied
        to the predictor data (and the combined dataset too).

        For a reference implementation of a custom predictor categorization,
        refer to `pdstools.utils.cdh_utils.default_predictor_categorization`.

        Parameters
        ----------
        df : Optional[pl.LazyFrame], optional
            A Polars Lazyframe to apply the categorization to.
            If not provided, applies it over the predictor data and combined datasets.
            By default, None
        categorization : Union[pl.Expr, Callable[..., pl.Expr]]
            A polars Expression (or method that returns one) to apply the mapping with.
            Should be based on Polars' when.then.otherwise syntax.
            By default, `pdstools.utils.cdh_utils.default_predictor_categorization`

        See also
        --------
        pdstools.utils.cdh_utils.default_predictor_categorization : The default method

        Examples
        --------
        >>> dm = ADMDatamart(my_data) #uses the OOTB predictor categorization

        >>> dm.apply_predictor_categorization(categorization=pl.when(
        >>> pl.col("PredictorName").cast(pl.Utf8).str.contains("Propensity")
        >>> ).then(pl.lit("External Model")
        >>> ).otherwise(pl.lit("Adaptive Model)")

        >>> # Now, every subsequent plot will use the custom categorization
        """

        categorization_expr: pl.Expr = (
            categorization() if callable(categorization) else categorization
        )

        if df is not None:
            return df.with_columns(PredictorCategory=categorization_expr)

        if hasattr(self, "predictor_data") and self.predictor_data is not None:
            self.predictor_data = self.predictor_data.with_columns(
                PredictorCategory=categorization_expr
            )
        if hasattr(self, "combined_data") and self.combined_data is not None:
            self.combined_data = self.combined_data.with_columns(
                PredictorCategory=categorization_expr
            )

    def save_data(
        self,
        path: Union[os.PathLike, str] = ".",
        selected_model_ids: Optional[List[str]] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Caches model_data and predictor_data to files.

        Parameters
        ----------
        path : str
            Where to place the files
        selected_model_ids : List[str]
            Optional list of model IDs to restrict to

        Returns
        -------
        (Optional[Path], Optional[Path]):
            The paths to the model and predictor data files
        """
        abs_path = Path(path).resolve()
        time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]
        modeldata_cache, predictordata_cache = None, None
        if self.model_data is not None:
            if selected_model_ids is None:
                modeldata_cache = pega_io.cache_to_file(
                    self.model_data, abs_path, name=f"cached_model_data_{time}"
                )
            else:
                modeldata_cache = pega_io.cache_to_file(
                    self.model_data.filter(pl.col("ModelID").is_in(selected_model_ids)),
                    abs_path,
                    name=f"cached_model_data_{time}",
                )
        if self.predictor_data is not None:
            if selected_model_ids is None:
                predictordata_cache = pega_io.cache_to_file(
                    self.predictor_data, abs_path, name=f"cached_predictor_data_{time}"
                )
            else:
                predictordata_cache = pega_io.cache_to_file(
                    self.predictor_data.filter(
                        pl.col("ModelID").is_in(selected_model_ids)
                    ),
                    abs_path,
                    name=f"cached_predictor_data_{time}",
                )

        return modeldata_cache, predictordata_cache

    @cached_property
    def unique_channels(self):
        """A consistently ordered set of unique channels in the data

        Used for making the color schemes in different plots consistent
        """
        return set(
            self.model_data.select(pl.col("Channel").unique().sort()).collect()[
                "Channel"
            ]
        )

    @cached_property
    def unique_configurations(self):
        """A consistently ordered set of unique configurations in the data

        Used for making the color schemes in different plots consistent
        """
        return set(
            self.model_data.select(pl.col("Configuration").unique())
            .collect()["Configuration"]
            .to_list()
        )

    @cached_property
    def unique_channel_direction(self):
        """A consistently ordered set of unique channel+direction combos in the data
        Used for making the color schemes in different plots consistent
        """
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
        """A consistently ordered set of unique configuration+channel+direction
        Used for making the color schemes in different plots consistent
        """
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
        """A consistently ordered set of unique predictor categories in the data
        Used for making the color schemes in different plots consistent
        """
        return set(
            self.predictor_data.select(
                pl.col("PredictorCategory").unique().sort()
            ).collect()["PredictorCategory"]
        )

from __future__ import annotations

__all__ = ["ADMDatamart"]

import datetime
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import polars as pl
import polars.selectors as cs

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
    2. Use one of the class methods: `from_ds_export`, `from_s3`, `from_dataflow_export` etc.

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
    first_action_dates: Optional[pl.LazyFrame]

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

        model_data_validated = self._validate_model_data(
            model_df, extract_pyname_keys=extract_pyname_keys
        )

        # First occurence of actions (before filtering!) kept here so we can derive the "New Actions"
        self.first_action_dates = self._get_first_action_dates(model_data_validated)

        self.model_data = cdh_utils._apply_query(model_data_validated, query)

        # TODO @stijn how do we ensure the model IDs intersect, also if there is a query argument?
        self.predictor_data = self._validate_predictor_data(predictor_df)

        self.combined_data = self.aggregates._combine_data(
            self.model_data, self.predictor_data
        )
        self.bin_aggregator = BinAggregator(dm=self)

    def _get_first_action_dates(self, df: Optional[pl.LazyFrame]) -> pl.LazyFrame:
        if df is None:
            return df
        # very_first_date = df.select(pl.col("SnapshotTime").min()).collect().item()
        return (
            df.group_by("Name")
            .agg(FirstSnapshotTime=pl.col("SnapshotTime").min())
            # .with_columns(
            #     pl.when(pl.col("FirstSnapshotTime") > very_first_date).then("FirstSnapshotTime")
            # )
            .sort("Name")
            # .filter(
            #     pl.col("FirstSnapshotTime") > very_first_date
            # )
        )

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

    @classmethod
    def from_pdc(cls, df: pl.LazyFrame, return_df=False):
        pdc_data = cdh_utils._read_pdc(df)

        adm_data = (
            pdc_data.filter(pl.col("ModelType") == "AdaptiveModel")
            .with_columns(
                pl.col("Performance").cast(pl.Float32),
                # pl.col("Positives").cast(pl.Float64),
                pl.col("Negatives").cast(pl.Float64),
                pl.col("TotalPositives").cast(pl.Float64),
                pl.col("TotalResponses").cast(pl.Float64),
                # pl.col("ResponseCount").cast(pl.Float64),
                pyTotalPredictors=pl.lit(None),
                pyActivePredictors=pl.lit(None),
                # CTR=(pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives"))),
                # ElapsedDays=(pl.col("Day").max() - pl.col("Day")).dt.total_days(),
                # see US-648869 and related items on the model technique
                pyModelTechnique=pl.when(
                    pl.col("ADMModelType").is_in(["GRADIENT_BOOST", "GradientBoost"])
                )
                .then(pl.lit("GradientBoost"))
                .otherwise(pl.lit("NaiveBayes")),
            )
            .rename(
                {
                    "ModelClass": "pyAppliesToClass",
                    "ModelID": "pyModelID",
                    "ModelName": "pyConfigurationName",
                    # BUG-875332: TotalResponses = Negatives + TotalPositives
                    # won't be fixed but is confusing
                    "Negatives": "pyNegatives",
                    "TotalPositives": "pyPositives",
                    "TotalResponses": "pyResponseCount",
                    "SnapshotTime": "pySnapshotTime",
                    "Performance": "pyPerformance",
                }
            )
            .with_columns(
                [
                    (
                        pl.col(c).alias(f"py{c}")
                        if c in pdc_data.collect_schema().names()
                        else pl.lit("").alias(f"py{c}")
                    )
                    for c in [
                        "Channel",
                        "Direction",
                        "Issue",
                        "Group",
                        "Name",
                        "Treatment",
                    ]
                ]
            )
            .drop(
                [
                    # "TotalResponses",
                    # "TotalPositives",
                    "ResponseCount",
                    "Positives",
                    "ModelType",
                    "ADMModelType",
                ]
                + [
                    c
                    for c in [
                        "pxObjClass",
                        "pzInsKey",
                        "Channel",
                        "Direction",
                        "Issue",
                        "Group",
                        "Name",
                        "Treatment",
                    ]
                    if c in pdc_data.collect_schema().names()
                ]
            )
        )

        if return_df:
            return adm_data

        return ADMDatamart(model_df=adm_data, extract_pyname_keys=True)

    def _validate_model_data(
        self,
        df: Optional[pl.LazyFrame],
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

        if not schema.get("SnapshotTime").is_temporal():  # pl.Datetime
            df = df.with_columns(
                SnapshotTime=cdh_utils.parse_pega_date_time_formats()
            ).sort("SnapshotTime", "ModelID")
        else:
            df = df.with_columns(pl.col("SnapshotTime").cast(pl.Datetime)).sort(
                "SnapshotTime", "ModelID"
            )

        df = df.with_columns(
            SuccessRate=(pl.col("Positives") / pl.col("ResponseCount")).fill_nan(
                pl.lit(0)
            ),
            IsUpdated=(
                (pl.col("ResponseCount").diff(1) != 0)
                | (pl.col("Positives").diff(1) != 0)
            )
            .fill_null(True)
            .over("ModelID"),
        ).with_columns(
            LastUpdate=pl.when("IsUpdated")
            .then("SnapshotTime")
            .forward_fill()
            .over("ModelID"),
        )

        df = cdh_utils._apply_schema_types(df, Schema.ADMModelSnapshot)

        return df

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
        if not schema.get("SnapshotTime").is_temporal():  # pl.Datetime
            df = df.with_columns(SnapshotTime=cdh_utils.parse_pega_date_time_formats())

        if "PredictorCategory" not in schema.names():
            df = self.apply_predictor_categorization(
                df=df
            )  # actual categorization not passed in?
        df = cdh_utils._apply_schema_types(df, Schema.ADMPredictorBinningSnapshot)
        return df

    def apply_predictor_categorization(
        self,
        categorization: Union[
            pl.Expr, Callable[..., pl.Expr], Dict[str, Union[str, List[str]]]
        ] = cdh_utils.default_predictor_categorization,
        *,
        use_regexp: bool = False,
        df: Optional[pl.LazyFrame] = None,
    ):
        """Apply a new predictor categorization to the datamart tables

        In certain plots, we use the predictor categorization to indicate what 'kind'
        a certain predictor is, such as IH, Customer, etc. Call this method with a
        custom Polars Expression (or a method that returns one) or a simple mapping
        and it will be applied to the predictor data (and the combined dataset too).
        When the categorization provides no match, the existing categories are kept
        as they are.

        For a reference implementation of a custom predictor categorization,
        refer to `pdstools.utils.cdh_utils.default_predictor_categorization`.

        Parameters
        ----------
        categorization : Union[pl.Expr, Callable[..., pl.Expr], Dict[str, Union[str, List[str]]]]
            A Polars Expression (or method that returns one) that returns the
            predictor categories. Should be based on Polars' when.then.otherwise syntax.
            Alternatively can be a dictionary of categories to (list of) string matches
            which can be either exact (the default) or regular expressions.
            By default, `pdstools.utils.cdh_utils.default_predictor_categorization` is used.
        use_regexp: bool, optional
            Treat the mapping patterns in the `categorization` dictionary as regular expressions
            rather than plain strings. When treated as regular expressions, they will be
            interpreted in non-strict mode, so invalid expressions will return in no match. See
            https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.str.contains.html
            for exact behavior of the regular expressions.
            By default, False
        df : Optional[pl.LazyFrame], optional
            A Polars Lazyframe to apply the categorization to.
            If not provided, applies it over the predictor data and combined datasets.
            By default, None

        See also
        --------
        pdstools.utils.cdh_utils.default_predictor_categorization : The default method

        Examples
        --------
        >>> dm = ADMDatamart(my_data) #uses the OOTB predictor categorization

        >>> # Uses a custom Polars expression to set the categories
        >>> dm.apply_predictor_categorization(categorization=pl.when(
        >>> pl.col("PredictorName").cast(pl.Utf8).str.contains("Propensity")
        >>> ).then(pl.lit("External Model")
        >>> )

        >>> # Uses a simple dictionary to set the categories
        >>> dm.apply_predictor_categorization(categorization={
        >>> "External Model" : ["Score", "Propensity"]}
        >>> )

        """

        def categorization_dict_to_polars_expr(categorization):
            # Dynamically constructing when/otherwise expression
            # see https://stackoverflow.com/questions/78818920/how-to-generate-when-then-constructs-in-polars-dynamically
            expr = pl
            for key, values in categorization.items():
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    expr = expr.when(
                        pl.col("PredictorName")
                        .cast(pl.Utf8)
                        .str.contains(value, literal=not use_regexp, strict=False)
                    ).then(pl.lit(key))
            return expr

        categorization_expr: pl.Expr = None
        if isinstance(categorization, dict) and len(categorization) > 0:
            categorization_expr = categorization_dict_to_polars_expr(categorization)
        elif callable(categorization):
            categorization_expr = categorization()
        elif isinstance(categorization, pl.Expr):
            categorization_expr = categorization
        else:
            return df

        def set_categories(df):
            if "PredictorCategory" in df.collect_schema().names():
                predictor_mapping = (
                    df.filter(pl.col("EntryType") != "Classifier")
                    .select("PredictorName", "PredictorCategory")
                    .unique()
                    .with_columns(NewPredictorCategory=categorization_expr)
                    .with_columns(
                        PredictorCategory=pl.coalesce(
                            "NewPredictorCategory", "PredictorCategory"
                        )
                    )
                    .drop("NewPredictorCategory")
                )
                df = df.drop("PredictorCategory").join(
                    predictor_mapping, on="PredictorName", how="left"
                )
            else:
                predictor_mapping = (
                    df.filter(pl.col("EntryType") != "Classifier")
                    .select(pl.col("PredictorName").unique())
                    .with_columns(PredictorCategory=categorization_expr)
                )
                df = df.join(predictor_mapping, on="PredictorName", how="left")
            return df

        if df is not None:
            return set_categories(df)

        if hasattr(self, "predictor_data") and self.predictor_data is not None:
            self.predictor_data = set_categories(self.predictor_data)
        if hasattr(self, "combined_data") and self.combined_data is not None:
            self.combined_data = set_categories(self.combined_data)

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
            self.predictor_data.select(pl.col("PredictorCategory").unique().sort())
            .filter(pl.col("PredictorCategory").is_not_null())
            .collect()["PredictorCategory"]
        )

    # min and max score (sum of log odds) per model, plus some extra summary statistics per model
    @classmethod
    def _minMaxScoresPerModel(cls, bin_data: pl.LazyFrame) -> pl.LazyFrame:
        minMaxScoresPerPredictor = (
            bin_data.filter(pl.col("EntryType") != "Inactive")
            .sort("BinIndex")
            .group_by(["ModelID", "PredictorName", "EntryType"], maintain_order=True)
            .agg(
                totalPos=pl.col("BinPositives").sum(),
                totalNeg=pl.col("BinNegatives").sum(),
                logOdds=cdh_utils.log_odds_polars("BinPositives", "BinNegatives"),
            )
            .with_columns(
                pl.col("logOdds").list.min().alias("logOddsMin"),
                pl.col("logOdds").list.max().alias("logOddsMax"),
            )
            .drop("logOdds")
        )

        return (
            minMaxScoresPerPredictor.group_by("ModelID", maintain_order=True)
            .agg(
                nActivePredictors=(pl.col("EntryType") == "Active").sum(),
                classifierLogOffset=(
                    1.0 + pl.col.totalPos.filter(EntryType="Classifier").first()
                ).log()
                - (1.0 + pl.col.totalNeg.filter(EntryType="Classifier").first()).log(),
                sumMinLogOdds=pl.col.logOddsMin.filter(EntryType="Active").sum(),
                sumMaxLogOdds=pl.col.logOddsMax.filter(EntryType="Active").sum(),
            )
            .with_columns(
                score_min=(pl.col.classifierLogOffset + pl.col.sumMinLogOdds)
                / (1 + pl.col.nActivePredictors),
                score_max=(pl.col.classifierLogOffset + pl.col.sumMaxLogOdds)
                / (1 + pl.col.nActivePredictors),
            )
        )

    def active_ranges(
        self, model_ids: Optional[Union[str, List[str]]] = None
    ) -> pl.LazyFrame:
        """Calculate the active, reachable bins in classifiers.

        The classifiers exported by Pega contain (in certain product versions) more than
        the bins that can be reached given the current state of the predictors. This method
        first calculates the min and max score range from the predictor log odds, then maps
        that to the interval boundaries of the classifier(s) to find the min and max index.

        It returns a LazyFrame with the score min/max, the min/max index, as well as the
        AUC as reported in the datamart data, when calculated from the full range, and when
        calculated from the reachable bins only.

        This information can be used in the Health Check documents or when verifying the
        AUC numbers from the datamart.

        Parameters
        ----------
        model_ids : Optional[Union[str, List[str]]], optional
            An optional list of model id's, or just a single one, to report on. When
            not given, the information is returned for all models.

        Returns
        -------
        pl.LazyFrame
            A table with all the index and AUC information for all the models with the following fields:

            Model Identification:
            - ModelID - The unique identifier for the model

            AUC Metrics:
            - AUC_Datamart - The AUC value as reported in the datamart
            - AUC_FullRange - The AUC calculated from the full range of bins in the classifier
            - AUC_ActiveRange - The AUC calculated from only the active/reachable bins

            Classifier Information:
            - Bins - The total number of bins in the classifier
            - nActivePredictors - The number of active predictors in the model

            Log Odds Information (mostly for internal use):
            - classifierLogOffset - The log offset of the classifier (baseline log odds)
            - sumMinLogOdds - The sum of minimum log odds across all active predictors
            - sumMaxLogOdds - The sum of maximum log odds across all active predictors
            - score_min - The minimum score (normalized sum of log odds including classifier offset)
            - score_max - The maximum score (normalized sum of log odds including classifier offset)

            Active Range Information:
            - idx_min - The minimum bin index that can be reached given the current binning of all predictors
            - idx_max - The maximum bin index that can be reached given the current binning of all predictors

        """
        import numpy as np

        def find_binindex(bounds, score):
            if len(bounds) == 1:
                return 0
            # Polars has search_sorted but it seems it doesn't do
            # the exact same thing as numpy searchsorted. We could
            # also use python's native bisect but that seems slower.
            return min(
                max(1, np.searchsorted(bounds, score, side="right").item()), len(bounds)
            )

        def auc_from_active_bins(pos, neg, idx_min, idx_max):
            return cdh_utils.auc_from_bincounts(
                pos[idx_min:idx_max], neg[idx_min:idx_max]
            )

        if isinstance(model_ids, str):
            query = pl.col("ModelID") == model_ids
        elif isinstance(model_ids, list):
            query = pl.col("ModelID").is_in(model_ids)
        else:
            query = None

        # Only take last binning snapshot per model ID. Usually there only is
        # one but it is configurable to have multiple. Also, sometimes the snapshot
        # time is null, due to import/export woes. This is problematic
        # for model data but here we just test for it and assume one snapshot.
        most_recent_binning_data = cdh_utils._apply_query(
            self.predictor_data.filter(
                (
                    # TODO consider using the "last" function of the aggregates
                    # last("predictor_data") instead of this, but that currently
                    # doesn't do that per Model ID. Probably should.
                    (pl.col("SnapshotTime").n_unique() == 1)
                    | (pl.col("SnapshotTime") == pl.col("SnapshotTime").max())
                ).over("ModelID")
            ),
            query,
            allow_empty=True,
        )

        classifier_info = (
            most_recent_binning_data.filter(EntryType="Classifier")
            .sort("BinIndex")
            .group_by("ModelID", maintain_order=True)
            .agg(
                AUC_Datamart=pl.col("Performance").first(),
                Bins=pl.len(),
                classifierBounds=pl.col("BinLowerBound").cast(pl.Float64),
                classifierPos=pl.col("BinPositives").cast(pl.Float64),
                classifierNeg=pl.col("BinNegatives").cast(pl.Float64),
            )
        )

        scores = self._minMaxScoresPerModel(most_recent_binning_data)

        return (
            classifier_info.join(scores, on="ModelID", how="left")
            .with_columns(
                AUC_FullRange=pl.map_groups(
                    exprs=[
                        pl.col("classifierPos").explode(),
                        pl.col("classifierNeg").explode(),
                    ],
                    function=lambda data: cdh_utils.auc_from_bincounts(
                        data[0], data[1]
                    ),
                    return_dtype=pl.Float64,
                    returns_scalar=True,
                ).over("ModelID"),
                idx_min=(
                    pl.map_groups(
                        exprs=[
                            pl.col("classifierBounds"),
                            pl.col("score_min"),
                        ],
                        function=lambda data: find_binindex(
                            data[0].to_list()[0],
                            data[1].item(),
                        ),
                        return_dtype=pl.Int32,
                        returns_scalar=True,
                    )
                    - 1
                ).over("ModelID"),
                idx_max=pl.map_groups(
                    exprs=[
                        pl.col("classifierBounds"),
                        pl.col("score_max"),
                    ],
                    function=lambda data: find_binindex(
                        data[0].to_list()[0],
                        data[1].item(),
                    ),
                    return_dtype=pl.Int32,
                    returns_scalar=True,
                ).over("ModelID"),
            )
            .with_columns(
                AUC_ActiveRange=pl.map_groups(
                    exprs=[
                        pl.col("classifierPos"),
                        pl.col("classifierNeg"),
                        pl.col("idx_min"),
                        pl.col("idx_max"),
                    ],
                    function=lambda data: auc_from_active_bins(
                        data[0].to_list()[0],
                        data[1].to_list()[0],
                        data[2].item(),
                        data[3].item(),
                    ),
                    return_dtype=pl.Float64,
                    returns_scalar=True,
                ).over("ModelID"),
            )
            .sort("ModelID")
            .drop("classifierBounds", "classifierPos", "classifierNeg")
            .select(cs.starts_with("AUC"), ~cs.starts_with("AUC"))
            .select("ModelID", ~cs.starts_with("ModelID"))
        )

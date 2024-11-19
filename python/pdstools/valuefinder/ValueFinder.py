import os
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import polars as pl

from ..pega_io import read_dataflow_output, read_ds_export
from ..pega_io.File import cache_to_file
from ..utils import cdh_utils
from ..utils.types import QUERY
from . import Schema
from .Aggregates import Aggregates
from .Plots import Plots


class ValueFinder:
    """Analyze the Value Finder dataset for detailed insights"""

    def __init__(
        self,
        df: pl.LazyFrame,
        *,
        query: Optional[QUERY] = None,
        n_customers: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        self.df: pl.LazyFrame = cdh_utils._apply_schema_types(df, Schema.pyValueFinder)
        self.df = cdh_utils._polars_capitalize(self.df)
        self.df = cdh_utils._apply_query(self.df, query)

        self.set_threshold(threshold)
        self.n_customers: int = n_customers or int(
            self.df.select(pl.col("CustomerID").n_unique().cast(pl.UInt32))
            .collect()
            .item()
        )

        self.nbad_stages = [
            "Eligibility",
            "Applicability",
            "Suitability",
            "Arbitration",
        ]
        self.aggregates = Aggregates(self)
        self.plot = Plots(self)

    @classmethod
    def from_ds_export(
        cls,
        filename: Optional[str] = None,
        base_path: Union[os.PathLike, str] = ".",
        *,
        query: Optional[QUERY] = None,
        n_customers: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        df = read_ds_export(filename or "value_finder", base_path)
        if df is None:
            raise ValueError(f"Could not find {filename} in {base_path}")

        return cls(df=df, query=query, n_customers=n_customers, threshold=threshold)

    @classmethod
    def from_dataflow_export(
        cls,
        files: Union[Iterable[str], str],
        *,
        query: Optional[QUERY] = None,
        n_customers: Optional[int] = None,
        threshold: Optional[float] = None,
        cache_file_prefix: str = "",
        extension: Literal["json"] = "json",
        compression: Literal["gzip"] = "gzip",
        cache_directory: Union[os.PathLike, str] = "cache",
    ):
        df = read_dataflow_output(
            files,
            cache_file_prefix + "value_finder",
            extension=extension,
            compression=compression,
            cache_directory=cache_directory,
        )

        return cls(
            df=df, query=query, n_customers=n_customers, threshold=threshold
        )  # pragma: no cover

    def set_threshold(self, new_threshold: Optional[float] = None):
        if new_threshold:
            self._th = pl.LazyFrame({"th": new_threshold})
        else:
            self._th = self.df.filter(pl.col("Stage") == "Eligibility").select(
                pl.quantile("ModelPropensity", 0.05).alias("th")
            )

    @cached_property
    def threshold(self):
        return self._th.collect().item()

    def save_data(self, path: Union[os.PathLike, str] = ".") -> Optional[Path]:
        """Cache the pyValueFinder dataset to a Parquet file

        Parameters
        ----------
        path : str
            Where to place the file

        Returns
        -------
        (Optional[Path], Optional[Path]):
            The paths to the model and predictor data files
        """
        time = datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]
        cache_file = cache_to_file(
            self.df, path, name=f"cached_value_finder_data_{time}"
        )
        return cache_file

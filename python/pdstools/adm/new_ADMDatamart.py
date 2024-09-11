from __future__ import annotations

from typing import Iterable, Optional, Union

import polars as pl

from ..utils import cdh_utils
from ..utils.cdh_utils import _polars_capitalize
from .ADMTrees import AGB
from .Aggregates import Aggregates
from .Plots import Plots


class ADMDatamart:
    model_data: pl.LazyFrame
    predictor_data: pl.LazyFrame
    combined_data: pl.LazyFrame

    def __init__(
        self,
        model_df: pl.LazyFrame,
        predictor_df: pl.LazyFrame,
        query: Optional[Union[pl.Expr, Iterable[pl.Expr]]] = None,
        extract_pyname_keys: bool = True,
    ) -> None:
        self.model_data = self._validate_model_data(
            model_df, query=query, extract_pyname_keys=extract_pyname_keys
        )
        self.predictor_data = self._validate_predictor_data(predictor_df)

        self.plot = Plots(datamart=self)
        self.aggregates = Aggregates(datamart=self)  # TODO: Final naming
        self.agb = AGB(datamart=self)

        self.combined_data = self.aggregates._combine_data(
            self.model_data, self.predictor_data
        )

    @classmethod
    def from_ds_export(cls): ...

    @classmethod
    def from_s3(cls): ...

    def _validate_model_data(
        self,
        df: pl.LazyFrame,
        query: Optional[Union[pl.Expr, Iterable[pl.Expr]]] = None,
        extract_pyname_keys: bool = True,
    ) -> pl.LazyFrame:
        df = _polars_capitalize(df)
        if extract_pyname_keys:
            df = cdh_utils._extract_keys(df)

        return df if not query else df.filter(query)

    def _validate_predictor_data(self, predictor_df: pl.LazyFrame) -> pl.LazyFrame:
        return predictor_df

    def generate_health_check(self): ...

    def generate_model_report(self): ...

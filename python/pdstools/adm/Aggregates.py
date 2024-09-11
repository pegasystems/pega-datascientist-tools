from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .new_ADMDatamart import ADMDatamart


class Aggregates:
    def __init__(self, datamart: "ADMDatamart"):
        self.datamart = datamart

    def last(self, data: pl.LazyFrame): ...

    def _combine_data(
        self, model_df: pl.LazyFrame, predictor_df: pl.LazyFrame
    ) -> pl.LazyFrame:
        return self.last(model_df).join(self.last(predictor_df), on="ModelID")

import polars as pl

from .Aggregates import Aggregates
from .Plots import Plots


class IH:
    data: pl.LazyFrame

    def __init__(self, data: pl.LazyFrame):
        self.data = data

        self.aggregates = Aggregates(ih=self)
        self.plots = Plots(ih=self)

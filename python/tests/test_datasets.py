"""
Testing the functionality of the built-in datasets
"""

import polars as pl
from pdstools import datasets


@pl.api.register_lazyframe_namespace("shape")
class Shape:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (ldf.select(pl.first().len()).collect().item(), len(ldf.columns))


def test_import_CDHSample():
    Sample = datasets.cdh_sample()
    assert Sample.model_data.shape == (1047, 28)


def test_import_SampleTrees():
    datasets.sample_trees()


def test_import_SampleValueFinder():
    vf = datasets.sample_value_finder()
    assert vf.df.shape == (27133, 98)

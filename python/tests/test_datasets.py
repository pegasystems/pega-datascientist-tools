"""
Testing the functionality of the built-in datasets
"""

import sys
import pathlib
basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import datasets
import polars as pl

@pl.api.register_lazyframe_namespace("shape")
class Shape:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (ldf.select(pl.first().count()).collect().item(), len(ldf.columns))


def test_import_CDHSample():
    Sample = datasets.CDHSample()
    assert Sample.modelData.shape == (1047,15)

def test_import_SampleTrees():
    datasets.SampleTrees()

def test_import_SampleValueFinder():
    vf = datasets.SampleValueFinder()
    assert vf.df.shape == (27133, 11)
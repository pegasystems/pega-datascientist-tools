"""
Testing the functionality of some end-to-end scenarios
"""

import pathlib

import polars as pl
import pytest
from plotly.graph_objs._figure import Figure
from pdstools import ADMDatamart

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def sample():
    return ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
    )


@pl.api.register_lazyframe_namespace("shape")
class Shape:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (
            ldf.select(pl.first().len()).collect().item(),
            len(ldf.collect_schema().names()),
        )


def test_end_to_end(sample: ADMDatamart):
    assert sample.model_data.shape == (1047, 28)

    assert sample.predictor_data.shape == (70735, 39)

    assert sample.combined_data.shape == (4576, 66)

    assert sample.aggregates.last().shape == (68, 28)
    assert sample.aggregates.last(table="predictor_data").shape == (4576, 39)
    assert sample.model_data.collect_schema()["SnapshotTime"] == pl.Datetime

    assert sample.context_keys == [
        "Channel",
        "Direction",
        "Issue",
        "Group",
        "Name",
        "Treatment",
    ]

    assert isinstance(sample.plot.bubble_chart(), Figure)
    assert (
        len(sample.plot.bubble_chart().data[0].x) == sample.aggregates.last().shape[0]
    )

    query = (pl.col("ResponseCount") > 500) & (pl.col("Group") == "CreditCards")
    queried = len(sample.aggregates.last().filter(query).collect())
    assert queried == 19
    assert len(sample.plot.bubble_chart(query=query).data[0].x) == queried

"""Testing the functionality of some end-to-end scenarios"""

import pathlib

import polars as pl
import pytest
from pdstools import ADMDatamart
from plotly.graph_objs._figure import Figure

basePath = pathlib.Path(__file__).parent.parent.parent.parent


@pytest.fixture
def sample():
    return ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
    )


def test_end_to_end(sample: ADMDatamart):
    assert sample.model_data.collect().shape == (1535, 30)

    assert sample.predictor_data.collect().shape == (70735, 39)

    assert sample.combined_data.collect().shape == (4576, 68)

    assert sample.aggregates.last().collect().shape == (98, 30)
    assert sample.aggregates.last(table="predictor_data").collect().shape == (4576, 39)
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
    assert len(sample.plot.bubble_chart().data[0].x) == sample.aggregates.last().collect().height

    query = (pl.col("ResponseCount") > 500) & (pl.col("Group") == "CreditCards")
    queried = len(sample.aggregates.last().filter(query).collect())
    assert queried == 24
    assert len(sample.plot.bubble_chart(query=query).data[0].x) == queried

"""Testing the functionality of some end-to-end scenarios"""

import polars as pl
from pdstools import ADMDatamart
from plotly.graph_objs._figure import Figure


def test_end_to_end(sample: ADMDatamart):
    assert sample.model_data.collect().shape == (131845, 20)

    assert sample.predictor_data.collect().shape == (246470, 28)

    assert sample.combined_data.collect().shape == (246470, 47)

    assert sample.aggregates.last().collect().shape == (864, 20)
    assert sample.aggregates.last(table="predictor_data").collect().shape == (246470, 28)
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

    query = (pl.col("ResponseCount") > 500) & (pl.col("Group") == "Loans")
    queried = len(sample.aggregates.last().filter(query).collect())
    assert queried == 77
    assert len(sample.plot.bubble_chart(query=query).data[0].x) == queried

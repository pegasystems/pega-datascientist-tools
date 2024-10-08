"""
Testing the functionality of some end-to-end scenarios
"""
import sys
import pathlib

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
import polars as pl
from pdstools import ADMDatamart
from plotly.graph_objs._figure import Figure


@pl.api.register_lazyframe_namespace("shape")
class Shape:
    """Get the shape of a lazy dataframe.

    This is just for testing purposes. This will break the computation graph.
    For testing it's convenient though because we can use shape interchangeably
    for both dataframes as well as lazyframes.
    """

    def __new__(cls, ldf: pl.LazyFrame):
        return (ldf.select(pl.first().len()).collect().item(), len(ldf.collect_schema().names()))


def test_end_to_end():
    datamart = ADMDatamart(f"{basePath}/data", timestamp_fmt="%Y-%m-%d %H:%M:%S")
    assert datamart.modelData.shape == (1047, 15)
    modelcols = {
        "ModelID",
        "Issue",
        "Group",
        "Channel",
        "Direction",
        "Last_Positives",
        "Last_ResponseCount",
        "Name",
        "Positives",
        "Configuration",
        "ResponseCount",
        "SnapshotTime",
        "Performance",
        "SuccessRate",
        "Treatment",
    }
    assert set(datamart.modelData.columns) == modelcols

    assert datamart.predictorData.shape == (70735, 19)
    predcols = {
        "ModelID",
        "Positives",
        "ResponseCount",
        "SnapshotTime",
        "PredictorName",
        "PredictorCategory",
        "Performance",
        "EntryType",
        "BinSymbol",
        "BinIndex",
        "BinType",
        "BinPositives",
        "BinNegatives",
        "BinResponseCount",
        "Type",
        "BinPropensity",
        "BinAdjustedPropensity",
        "Contents",
        "PredictorCategory",
        "GroupIndex",
    }
    assert set(datamart.predictorData.columns) == predcols

    assert datamart.combinedData.shape == (4576, 33)
    assert set(datamart.combinedData.columns) == modelcols.union(predcols).union(
        {"PerformanceBin", "PositivesBin", "ResponseCountBin", "SnapshotTimeBin"}
    )

    assert datamart.last().shape == (68, 15)
    assert datamart.last("predictorData").shape == (4576, 19)
    assert datamart.modelData.schema["SnapshotTime"] == pl.Datetime

    assert datamart.context_keys == ["Channel", "Direction", "Issue", "Group"]
    assert datamart.missing_model == {
        "BinIndex",
        "Type",
        "PredictorName",
        "PredictorCategory",
        "BinSymbol",
        "BinType",
        "BinPositives",
        "BinResponseCount",
        "EntryType",
        "BinNegatives",
        "Contents",
        "GroupIndex",
    }

    assert datamart.missing_preds == {
        "Group",
        "Issue",
        "Channel",
        "Direction",
        "Name",
        "Configuration",
        "Treatment",
        "PredictorCategory",
    }

    assert datamart.renamed_model == {
        "Configuration",
        "ModelID",
        "Treatment",
        "SnapshotTime",
        "Name",
        "Group",
        "Performance",
        "Direction",
        "Positives",
        "ResponseCount",
        "Issue",
        "Channel",
    }

    assert datamart.renamed_preds == {
        "Performance",
        "ModelID",
        "EntryType",
        "BinIndex",
        "Type",
        "BinPositives",
        "BinType",
        "BinNegatives",
        "Positives",
        "ResponseCount",
        "BinSymbol",
        "SnapshotTime",
        "BinResponseCount",
        "PredictorName",
        "Contents",
        "GroupIndex",
    }

    assert isinstance(datamart.plotPerformanceSuccessRateBubbleChart(), Figure)
    assert (
        len(datamart.plotPerformanceSuccessRateBubbleChart().data[0].x)
        == datamart.last().shape[0]
    )

    query = (pl.col("ResponseCount") > 500) & (pl.col("Group") == "CreditCards")
    queried = len(datamart.last().filter(query))
    assert queried == 19
    assert (
        len(datamart.plotPerformanceSuccessRateBubbleChart(query=query).data[0].x)
        == queried
    )

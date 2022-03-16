import sys

sys.path.append("python")
from cdhtools import ADMDatamart

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from plotly.graph_objs._figure import Figure


def test_end_to_end():
    datamart = ADMDatamart("data")
    assert datamart.modelData.shape == (1047, 12)
    modelcols = {
        "ModelID",
        "Issue",
        "Group",
        "Channel",
        "Direction",
        "ModelName",
        "Positives",
        "Configuration",
        "ResponseCount",
        "SnapshotTime",
        "Performance",
        "SuccessRate",
    }
    assert set(datamart.modelData.columns) == modelcols

    assert datamart.predictorData.shape == (70735, 19)
    predcols = {
        "ModelID",
        "Positives",
        "ResponseCount",
        "SnapshotTime",
        "PredictorName",
        "Performance",
        "EntryType",
        "BinSymbol",
        "BinIndex",
        "BinType",
        "BinPositives",
        "BinNegatives",
        "BinResponseCount",
        "Type",
        "BinPositivesPercentage",
        "BinNegativesPercentage",
        "BinResponseCountPercentage",
        "BinPropensity",
        "BinAdjustedPropensity",
    }
    assert set(datamart.predictorData.columns) == predcols

    assert datamart.combinedData.shape == (4865, 30)
    assert set(datamart.combinedData.columns) == modelcols.union(predcols).union(
        {"PerformanceBin", "PositivesBin", "ResponseCountBin", "SnapshotTimeBin"}
    )

    assert datamart.last().shape == (70, 12)
    assert datamart.last("predictorData").shape == (4865, 19)

    assert all(
        item == "object"
        for item in list(
            datamart.modelData[
                ["Issue", "Group", "Channel", "Direction", "ModelName"]
            ].dtypes
        )
    )
    assert datamart.modelData.Positives.dtype == "int64"
    assert is_datetime(datamart.modelData.SnapshotTime.dtype)

    assert datamart.facets == ["Channel", "Direction", "Issue", "Group"]
    assert datamart.missing_model == [
        "PredictorName",
        "EntryType",
        "BinSymbol",
        "BinIndex",
        "BinType",
        "BinPositives",
        "BinNegatives",
        "BinResponseCount",
        "Type",
        "BinPositivesPercentage",
        "BinNegativesPercentage",
        "BinResponseCountPercentage",
    ]

    assert datamart.missing_preds == [
        "Issue",
        "Group",
        "Channel",
        "Direction",
        "ModelName",
        "Configuration",
    ]

    assert datamart.renamed_model == {
        "ModelID": "ModelID",
        "Issue": "Issue",
        "Group": "Group",
        "Channel": "Channel",
        "Direction": "Direction",
        "Name": "ModelName",
        "Positives": "Positives",
        "ConfigurationName": "Configuration",
        "ResponseCount": "ResponseCount",
        "SnapshotTime": "SnapshotTime",
        "Performance": "Performance",
    }

    assert datamart.renamed_preds == {
        "ModelID": "ModelID",
        "Positives": "Positives",
        "ResponseCount": "ResponseCount",
        "SnapshotTime": "SnapshotTime",
        "PredictorName": "PredictorName",
        "Performance": "Performance",
        "EntryType": "EntryType",
        "BinSymbol": "BinSymbol",
        "BinIndex": "BinIndex",
        "BinType": "BinType",
        "BinPositives": "BinPositives",
        "BinNegatives": "BinNegatives",
        "BinResponseCount": "BinResponseCount",
        "Type": "Type",
        "BinPositivesPercentage": "BinPositivesPercentage",
        "BinNegativesPercentage": "BinNegativesPercentage",
        "BinResponseCountPercentage": "BinResponseCountPercentage",
    }

    assert type(datamart.plotPerformanceSuccessRateBubbleChart()) == Figure
    assert (
        len(datamart.plotPerformanceSuccessRateBubbleChart().data[0].x)
        == datamart.last().shape[0]
    )

    query = 'ResponseCount > 500 and Group == "CreditCards"'
    queried = len(datamart.last().query(query))
    assert queried == 19
    assert (
        len(datamart.plotPerformanceSuccessRateBubbleChart(query=query).data[0].x)
        == queried
    )
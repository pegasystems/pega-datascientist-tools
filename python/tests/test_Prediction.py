"""
Testing the functionality of the Prediction class
"""

import datetime
import sys
import pytest
import polars as pl
import pathlib

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import Prediction, cdh_utils, errors

mock_prediction_data = pl.DataFrame(
    {
        "SnapshotTime": datetime.datetime.now(),
        "ModelClass": ["DATA-DECISION-REQUEST-CUSTOMER"] * 8,
        "ModelName": ["PREDICTWEBPROPENSITY"] * 4 + ["PREDICTMOBILEPROPENSITY"] * 4,
        "Channel": ["Web"] * 4 + ["Mobile"] * 4,
        "Direction": ["Inbound"] * 4 + ["Outbound"] * 4,
        "ModelType": [
            "Prediction_Control",
            "Prediction_Test",
            "Prediction_NBA",
            "Prediction",
            "Prediction_Control",
            "Prediction_Test",
            "Prediction_NBA",
            "Prediction",
        ],
        "Positives": [100, 400, 500, 1000, 200, 800, 1000, 2000],
        "Negatives": [1000, 2000, 3000, 6000, 3000, 6000, 9000, 18000],
        "ResponseCount": [1100, 2400, 3500, 7000, 3200, 6800, 10000, 20000],
        "Performance": [0.65] * 4 + [0.70] * 4,
    }
).lazy()


@pytest.fixture
def test():
    """Fixture to serve as class to call functions from."""
    return Prediction(mock_prediction_data)


@pytest.fixture
def test2(test):
    today = datetime.datetime.now()
    return Prediction(
        pl.concat(
            [
                mock_prediction_data.with_columns(SnapshotTime=pl.lit(today)),
                mock_prediction_data.with_columns(
                    SnapshotTime=pl.lit(today + datetime.timedelta(days=-1))
                ),
            ],
            how="vertical",
        )
    )


def test_available(test):
    assert test.is_available


def test_summary_by_channel_cols(test):
    summary = test.summary_by_channel().collect()
    assert summary.columns == [
        "ModelName",
        "Channel",
        "Direction",
        "isStandardNBADPrediction",
        "isMultiChannelPrediction",
        "Lift",
        "Performance",
        "Positives",
        "Negatives",
        "ResponseCount",
        "Positives_Test",
        "Positives_Control",
        "Negatives_Test",
        "Negatives_Control",
        "usesImpactAnalyzer",
        "CTR",
        "isValidPrediction",
    ]
    assert len(summary) == 2


def test_summary_by_channel_channels(test):
    summary = test.summary_by_channel().collect()
    assert summary.select(pl.len()).item() == 2


def test_summary_by_channel_validity(test):
    summary = test.summary_by_channel().collect()
    assert summary["isValidPrediction"].to_list() == [True, True]

def test_summary_by_channel_ia(test):
    summary = test.summary_by_channel().collect()
    assert summary["usesImpactAnalyzer"].to_list() == [True, True]

def test_summary_by_channel_lift(test):
    summary = test.summary_by_channel().collect()
    assert [round(x, 5) for x in summary["Lift"].to_list()] == [0.88235, 0.83333]


def test_summary_by_channel_trend(test):
    summary = test.summary_by_channel(keep_trend_data=True).collect()
    assert summary.select(pl.len()).item() == 2


def test_summary_by_channel_trend2(test2):
    summary = test2.summary_by_channel(keep_trend_data=True).collect()
    assert summary.select(pl.len()).item() == 4


def test_overall_summary_cols(test):
    summary = test.overall_summary().collect()
    assert summary.columns == [
        "Number of Valid Channels",
        "Overall Lift",
        "Performance",
        "Positives",
        "ResponseCount",
        "Channel with Minimum Negative Lift",
        "Minimum Negative Lift",
        "usesImpactAnalyzer",
        "CTR",
    ]
    assert len(summary) == 1


def test_overall_summary_n_valid_channels(test):
    assert test.overall_summary().collect()["Number of Valid Channels"].item() == 2


def test_overall_summary_overall_lift(test):
    assert round(test.overall_summary().collect()["Overall Lift"].item(), 5) == 0.86964


def test_overall_summary_positives(test):
    assert test.overall_summary().collect()["Positives"].item() == 3000


def test_overall_summary_responsecount(test):
    assert test.overall_summary().collect()["ResponseCount"].item() == 27000


def test_overall_summary_channel_min_lift(test):
    assert (
        test.overall_summary().collect()["Channel with Minimum Negative Lift"].item()
        is None
    )


def test_overall_summary_min_lift(test):
    assert test.overall_summary().collect()["Minimum Negative Lift"].item() is None


def test_overall_summary_ctr(test):
    assert round(test.overall_summary().collect()["CTR"].item(), 5) == 0.11111


def test_overall_summary_ia(test):
    assert test.overall_summary().collect().select(pl.col("usesImpactAnalyzer")).item()

    test = Prediction(mock_prediction_data.filter(pl.col("ModelType") != "Prediction_NBA"))
    assert not test.overall_summary().collect().select(pl.col("usesImpactAnalyzer")).item()


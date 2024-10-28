"""
Testing the functionality of the Prediction class
"""

import datetime

import polars as pl
import pytest
from pdstools import Prediction, cdh_utils

mock_prediction_data = pl.DataFrame(
    {
        "pySnapShotTime": cdh_utils.to_prpc_date_time(datetime.datetime.now())[
            0:15
        ],  # Polars doesn't like time zones like GMT+0200
        "pyModelId": ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"] * 4
        + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTMOBILEPROPENSITY"] * 4,
        # "Channel": ["Web"] * 4 + ["Mobile"] * 4,
        # "Direction": ["Inbound"] * 4 + ["Outbound"] * 4,
        "pyModelType": "PREDICTION",
        "pySnapshotType": (["Daily"] * 3 + [None]) * 2,
        "pyDataUsage": [
            "Control",
            "Test",
            "NBA",
            "",
            "Control",
            "Test",
            "NBA",
            "",
        ],
        "pyPositives": [100, 400, 500, 1000, 200, 800, 1000, 2000],
        "pyNegatives": [1000, 2000, 3000, 6000, 3000, 6000, 9000, 18000],
        "pyCount": [1100, 2400, 3500, 7000, 3200, 6800, 10000, 20000],
        "pyValue": [0.65] * 4 + [0.70] * 4,
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
                mock_prediction_data.with_columns(
                    pySnapShotTime=pl.lit(cdh_utils.to_prpc_date_time(today)[0:15])
                ),
                mock_prediction_data.with_columns(
                    pySnapShotTime=pl.lit(
                        cdh_utils.to_prpc_date_time(
                            today + datetime.timedelta(days=-1)
                        )[0:15]
                    )
                ),
            ],
            how="vertical",
        )
    )


def test_available(test):
    assert test.is_available
    assert test.is_valid


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
        "Positives_NBA",
        "Negatives_Test",
        "Negatives_Control",
        "Negatives_NBA",
        "usesImpactAnalyzer",
        "ControlPercentage",
        "TestPercentage",
        "CTR",
        "isValid",
    ]
    assert len(summary) == 2


def test_summary_by_channel_channels(test):
    summary = test.summary_by_channel().collect()
    assert summary.select(pl.len()).item() == 2


def test_summary_by_channel_validity(test):
    summary = test.summary_by_channel().collect()
    assert summary["isValid"].to_list() == [True, True]


def test_summary_by_channel_ia(test):
    summary = test.summary_by_channel().collect()
    assert summary["usesImpactAnalyzer"].to_list() == [True, True]

    test = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (
                pl.col("pyModelId")
                == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"
            )
        )
    )
    assert test.summary_by_channel().collect()["usesImpactAnalyzer"].to_list() == [
        False,
        True,
    ]


def test_summary_by_channel_lift(test):
    summary = test.summary_by_channel().collect()
    assert [round(x, 5) for x in summary["Lift"].to_list()] == [0.88235, 0.83333]


def test_summary_by_channel_controlpct(test):
    summary = test.summary_by_channel().collect()
    assert [round(x, 5) for x in summary["ControlPercentage"].to_list()] == [
        16.0,
        15.71429,
    ]
    assert [round(x, 5) for x in summary["TestPercentage"].to_list()] == [
        34.0,
        34.28571,
    ]


def test_summary_by_channel_trend(test):
    summary = test.summary_by_channel(by_period="1d").collect()
    assert summary.select(pl.len()).item() == 2


def test_summary_by_channel_trend2(test2):
    summary = test2.summary_by_channel(by_period="1d").collect()
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
        "ControlPercentage",
        "TestPercentage",
        "CTR",
    ]
    assert len(summary) == 1


def test_overall_summary_n_valid_channels(test):
    print(test.overall_summary().collect())
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


def test_overall_summary_controlpct(test):
    assert (
        round(test.overall_summary().collect()["ControlPercentage"].item(), 5)
        == 15.92593
    )
    assert (
        round(test.overall_summary().collect()["TestPercentage"].item(), 5) == 34.07407
    )


def test_overall_summary_ia(test):
    assert test.overall_summary().collect().select(pl.col("usesImpactAnalyzer")).item()

    test = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (
                pl.col("pyModelId")
                == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"
            )
        )
    )
    assert test.overall_summary().collect()["usesImpactAnalyzer"].to_list() == [True]

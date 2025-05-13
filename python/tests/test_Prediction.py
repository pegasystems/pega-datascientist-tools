"""
Testing the functionality of the Prediction class
"""

import datetime

import polars as pl
import pytest
from pdstools import Prediction
from pdstools.utils import cdh_utils

mock_prediction_data = pl.DataFrame(
    {
        "pySnapShotTime": cdh_utils.to_prpc_date_time(datetime.datetime(2040, 4, 1))[
            0:15
        ],  # Polars doesn't like time zones like GMT+0200
        "pyModelId": ["DATA-DECISION-REQUEST-CUSTOMER!MYCUSTOMPREDICTION"] * 4
        + ["DATA-DECISION-REQUEST-CUSTOMER!PredictActionPropensity"] * 4
        + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTMOBILEPROPENSITY"] * 4
        + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"] * 4,
        "pyModelType": "PREDICTION",
        "pySnapshotType": (["Daily"] * 3 + [None]) * 4,
        "pyDataUsage": ["Control", "Test", "NBA", ""] * 4,
        "pyPositives": [100, 400, 500, 1000, 200, 800, 1000, 2000] * 2,
        "pyNegatives": [1000, 2000, 3000, 6000, 3000, 6000, 9000, 18000] * 2,
        "pyCount": [1100, 2400, 3500, 7000, 3200, 6800, 10000, 20000] * 2,
        "pyValue": ([0.65] * 4 + [0.70] * 4) * 2,
    }
).lazy()


@pytest.fixture
def preds_singleday():
    """Fixture to serve as class to call functions from."""
    return Prediction(mock_prediction_data)


@pytest.fixture
def preds_fewdays():
    return Prediction(
        pl.concat(
            [
                mock_prediction_data.with_columns(
                    pySnapShotTime=pl.lit(cdh_utils.to_prpc_date_time(datetime.datetime(2040, 5, 1))[0:15])
                ),
                mock_prediction_data.with_columns(
                    pySnapShotTime=pl.lit(
                        cdh_utils.to_prpc_date_time(
                            datetime.datetime(2040, 5, 16)
                        )[0:15]
                    )
                ),
            ],
            how="vertical",
        )
    )


def test_available(preds_singleday):
    print(preds_singleday.predictions.collect())
    assert preds_singleday.is_available
    assert preds_singleday.is_valid


def test_summary_by_channel_cols(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary.columns == [
        "Prediction",
        "Channel",
        "Direction",
        "isStandardNBADPrediction",
        "isMultiChannelPrediction",
        'DateRange Min',
        'DateRange Max',
        'Duration',
        "Performance",
        "Positives",
        "Negatives",
        "Responses",
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
        "CTR_Test",
        "CTR_Control",
        "CTR_NBA",
        "ChannelDirectionGroup",
        "isValid",
        "Lift",
    ]


def test_summary_by_channel_channels(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary.select(pl.len()).item() == 4


def test_summary_by_channel_validity(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary["isValid"].to_list() == [True, True, True, True]

def test_summary_by_channel_trend(preds_singleday):
    summary = preds_singleday.summary_by_channel(by_period="1d").collect()
    assert summary.select(pl.len()).item() == 4


def test_summary_by_channel_trend2(preds_fewdays):
    summary = preds_fewdays.summary_by_channel(by_period="1d").collect()
    assert summary.select(pl.len()).item() == 8



def test_summary_by_channel_ia(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert summary["usesImpactAnalyzer"].to_list() == [True, True, True, True]

    preds_singleday = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (
                pl.col("pyModelId")
                == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"
            )
        )
    )
    # only Web still has the NBA indicator
    assert preds_singleday.summary_by_channel().collect()["usesImpactAnalyzer"].to_list() == [
        False,
        False,
        False,
        True,
    ]


def test_summary_by_channel_lift(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert [round(x, 5) for x in summary["Lift"].to_list()] == [0.83333, 0.88235] * 2


def test_summary_by_channel_controlpct(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()
    assert [round(x, 5) for x in summary["ControlPercentage"].to_list()] == [
        15.71429,
        16.0,
    ] * 2
    assert [round(x, 5) for x in summary["TestPercentage"].to_list()] == [
        34.28571,
        34.0,
    ] * 2


def test_summary_by_channel_range(preds_fewdays):
    summary = preds_fewdays.summary_by_channel().collect()
    assert summary['DateRange Min'].to_list() == [datetime.date(2040,5,1)]*4
    assert summary['Positives'].to_list() == [2000,4000,2000,4000]

    summary = preds_fewdays.summary_by_channel(start_date=datetime.date(2040, 5, 15)).collect()
    assert summary['DateRange Min'].to_list() == [datetime.date(2040,5,16)]*4
    assert summary['Positives'].to_list() == [1000,2000,1000,2000]


def test_summary_by_channel_channeldirectiongroup(preds_singleday):
    summary = preds_singleday.summary_by_channel().collect()

    assert summary["isMultiChannelPrediction"].to_list() == [False, True, False, False]
    assert summary["isStandardNBADPrediction"].to_list() == [False, True, True, True]
    assert summary["ChannelDirectionGroup"].to_list() == [
        "Other",
        "Other",
        "Mobile/Inbound",
        "Web/Inbound",
    ]


def test_overall_summary_cols(preds_singleday):
    summary = preds_singleday.overall_summary().collect()
    assert summary.columns == [
        'DateRange Min',
        'DateRange Max',
        'Duration',
        "Number of Valid Channels",
        "Overall Lift",
        "Performance",
        "Positives Inbound",
        "Positives Outbound",
        "Responses Inbound",
        "Responses Outbound",
        "Channel with Minimum Negative Lift",
        "Minimum Negative Lift",
        "usesImpactAnalyzer",
        "ControlPercentage",
        "TestPercentage",
    ]
    assert len(summary) == 1


def test_overall_summary_n_valid_channels(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Number of Valid Channels"].item() == 3


def test_overall_summary_overall_lift(preds_singleday):
    # print(test.overall_summary().collect())
    # print(test.summary_by_channel().collect())
    assert round(preds_singleday.overall_summary().collect()["Overall Lift"].item(), 5) == 0.86217


def test_overall_summary_positives(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Positives Inbound"].item() == 3000
    assert preds_singleday.overall_summary().collect()["Positives Outbound"].item() == 0 # some channels unknown/multi-channel


def test_overall_summary_responsecount(preds_singleday):
    print (preds_singleday.summary_by_channel().select(['Channel','Direction','Responses']).collect())
    assert preds_singleday.overall_summary().collect()["Responses Inbound"].item() == 27000
    assert preds_singleday.overall_summary().collect()["Responses Outbound"].item() == 0 

def test_overall_summary_channel_min_lift(preds_singleday):
    assert (
        preds_singleday.overall_summary().collect()["Channel with Minimum Negative Lift"].item()
        is None
    )

def test_overall_summary_by_period(preds_fewdays):
    summ = preds_fewdays.overall_summary(by_period="1d").collect()
    assert summ.height == 2

def test_overall_summary_min_lift(preds_singleday):
    assert preds_singleday.overall_summary().collect()["Minimum Negative Lift"].item() is None


def test_overall_summary_controlpct(preds_singleday):
    assert (
        round(preds_singleday.overall_summary().collect()["ControlPercentage"].item(), 5)
        == 15.88235
    )
    assert (
        round(preds_singleday.overall_summary().collect()["TestPercentage"].item(), 5) == 34.11765
    )


def test_overall_summary_ia(preds_singleday):
    assert preds_singleday.overall_summary().collect().select(pl.col("usesImpactAnalyzer")).item()

    preds_singleday = Prediction(
        mock_prediction_data.filter(
            (pl.col("pyDataUsage") != "NBA")
            | (
                pl.col("pyModelId")
                == "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"
            )
        )
    )
    assert preds_singleday.overall_summary().collect()["usesImpactAnalyzer"].to_list() == [True]


def test_plots():
    prediction = Prediction.from_mock_data()

    assert prediction.plot.performance_trend() is not None
    assert prediction.plot.lift_trend() is not None
    assert prediction.plot.responsecount_trend() is not None
    assert prediction.plot.ctr_trend() is not None

    assert prediction.plot.performance_trend("1w") is not None
    assert isinstance(prediction.plot.lift_trend("2d", return_df=True), pl.LazyFrame)
    assert prediction.plot.responsecount_trend("1m") is not None
    assert prediction.plot.ctr_trend("5d") is not None

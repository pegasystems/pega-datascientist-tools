"""
Testing the functionality of the Prediction class
"""

import sys
import pytest
import zipfile
import polars as pl
from polars.testing import assert_frame_equal
import itertools
from pandas.errors import UndefinedVariableError
import pathlib

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import Prediction, cdh_utils, errors


@pytest.fixture
def test():
    """Fixture to serve as class to call functions from."""
    return Prediction(
        pl.DataFrame(
            {
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
                "Negatives": [100, 400, 500, 1000, 200, 800, 1000, 2000],
                "Positives": [1000, 2000, 3000, 6000, 3000, 6000, 9000, 18000],
                "ResponseCount": [1100, 2400, 3500, 7000, 3200, 6800, 10000, 20000],
                "Performance": [0.65] * 4 + [0.70] * 4,
            }
        ).lazy()
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
        "isValidPrediction",
        "Lift",
        "Performance",
        "Positives",
        "Negatives",
        "ResponseCount",
        "Positives_Test",
        "Positives_Control",
        "Negatives_Test",
        "Negatives_Control",
        "CTR",
    ]
    assert len(summary) == 2


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
        "CTR",
    ]
    assert len(summary) == 1

def test_overall_summary_n_valid_channels(test):
    assert test.overall_summary().collect()["Number of Valid Channels"].item() == 2

def test_overall_summary_overall_lift(test):
    assert round(test.overall_summary().collect()["Overall Lift"].item(), 5) == -0.06518

def test_overall_summary_positives(test):
    assert test.overall_summary().collect()["Positives"].item() == 24000

def test_overall_summary_responsecount(test):
    assert test.overall_summary().collect()["ResponseCount"].item() == 27000

def test_overall_summary_channel_min_lift(test):
    assert test.overall_summary().collect()["Channel with Minimum Negative Lift"].item() == "Web"

def test_overall_summary_min_lift(test):
    assert round(test.overall_summary().collect()["Minimum Negative Lift"].item(), 5) == -0.08333

def test_overall_summary_ctr(test):
    assert round(test.overall_summary().collect()["CTR"].item(), 5) == round(24000/27000, 5)

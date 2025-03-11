"""
Testing the functionality of the ADMDatamart Aggregates functions
"""

import pathlib

import pytest
from pdstools import ADMDatamart
from pdstools.adm.Aggregates import Aggregates

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def agg():
    """Fixture to serve as class to call functions from."""
    return Aggregates(
        ADMDatamart.from_ds_export(
            base_path=f"{basePath}/data",
            model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
            predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
        )
    )


def test_aggregates_init(agg):
    assert agg


def test_aggregate_model_summary(agg):
    assert agg.model_summary().collect().shape[0] == 68
    assert agg.model_summary().collect().shape[1] == 20


def test_aggregate_predictor_counts(agg):
    assert agg.predictor_counts().collect().shape[0] == 78
    assert agg.predictor_counts().collect().shape[1] == 5


# TODO expand testing, more variations etc, extra configs as arguments to the aggregator


def test_aggregate_summary_by_channel(agg):
    summary_by_channel = agg.summary_by_channel().collect()
    assert summary_by_channel.height == 3
    assert summary_by_channel.width == 22
    assert summary_by_channel["ChannelDirection"].to_list() == [
        "Email/Outbound",
        "SMS/Outbound",
        "Web/Inbound",
    ]
    assert summary_by_channel["ChannelDirectionGroup"].to_list() == [
        "E-mail/Outbound",
        "SMS/Outbound",
        "Web/Inbound",
    ]
    assert summary_by_channel["Actions"].to_list() == [24, 27, 19]
    assert summary_by_channel["Positives"].to_list() == [1181, 6367, 4638]
    assert summary_by_channel["Responses"].to_list() == [48603, 74054, 47177]
    assert summary_by_channel["Used Actions"].to_list() == [24, 27, 19]
    assert summary_by_channel["Treatments"].to_list() == [0] * 3
    assert summary_by_channel["Used Treatments"].to_list() == [0] * 3
    assert summary_by_channel["Duration"].to_list() == [18000] * 3


def test_aggregate_summary_by_channel_and_time(agg):
    summary_by_channel = agg.summary_by_channel(by_period="4h").collect()
    assert summary_by_channel.height == 6
    assert summary_by_channel.width == 23
    assert summary_by_channel["Responses"].to_list() == [
        27322,
        13062,
        45786,
        19557,
        28230,
        10890,
    ]


def test_aggregate_overall_summary(agg):
    overall_summary = agg.overall_summary().collect()
    assert overall_summary.height == 1
    assert overall_summary.width == 18
    assert overall_summary["Number of Valid Channels"].item() == 3
    assert overall_summary["Actions"].item() == 35
    assert overall_summary["Treatments"].item() == 0
    assert round(overall_summary["OmniChannel"].item(), 5) == 0.65331


def test_aggregate_overall_summary_by_time(agg):
    overall_summary = agg.overall_summary(by_period="1h").collect()
    print(overall_summary.select("Perdiod", "Responses", "Duration"))
    assert overall_summary.height == 4
    assert overall_summary.width == 19
    assert overall_summary["Number of Valid Channels"].item() == 3
    assert overall_summary["Actions"].item() == 35
    assert overall_summary["Treatments"].item() == 0
    assert round(overall_summary["OmniChannel"].item(), 5) == 0.65331


def test_summary_by_configuration(agg):
    configuration_summary = agg.summary_by_configuration().collect()
    assert "AGB" in configuration_summary.columns

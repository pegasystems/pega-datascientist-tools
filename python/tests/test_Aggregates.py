"""
Testing the functionality of the ADMDatamart Aggregates functions
"""

import pathlib

import pytest
from pdstools import ADMDatamart
from pdstools.adm.Aggregates import Aggregates
import polars as pl

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def dm_aggregates():
    """Fixture to serve as class to call functions from."""
    return Aggregates(
        ADMDatamart.from_ds_export(
            base_path=f"{basePath}/data",
            model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
            predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
        )
    )


def test_init(dm_aggregates):
    assert dm_aggregates


def test_model_summary(dm_aggregates):
    assert dm_aggregates.model_summary().collect().shape[0] == 68
    assert dm_aggregates.model_summary().collect().shape[1] == 20


def test_predictor_counts(dm_aggregates):
    assert dm_aggregates.predictor_counts().collect().shape[0] == 78
    assert dm_aggregates.predictor_counts().collect().shape[1] == 5


def test_summary_by_channel(dm_aggregates):
    summary_by_channel = dm_aggregates.summary_by_channel().collect()
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
    assert [round(x, 6) for x in summary_by_channel["OmniChannel"].to_list()] == [
        0.604167,
        0.592593,
        0.763158,
    ]
    assert summary_by_channel["isValid"].to_list() == [True, True, True]

    # Force one channel to be invalid
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Positives=pl.when(pl.col.Channel == "SMS")
        .then(pl.lit(0))
        .otherwise("Positives")
    )
    summary_by_channel = dm_aggregates.summary_by_channel().collect()
    assert summary_by_channel["isValid"].to_list() == [True, False, True]

    # Force only one channel to be valid
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Positives=pl.when(pl.col.Channel != "Web")
        .then(pl.lit(0))
        .otherwise("Positives")
    )
    summary_by_channel = dm_aggregates.summary_by_channel().collect()
    assert summary_by_channel["isValid"].to_list() == [False, False, True]
    assert summary_by_channel["OmniChannel"].to_list() == [
        None,
        None,
        0,
    ]


def test_aggregate_summary_by_channel_and_time(dm_aggregates):
    summary_by_channel = dm_aggregates.summary_by_channel(by_period="4h").collect()
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
    assert [round(x, 6) for x in summary_by_channel["OmniChannel"].to_list()] == [
        0.604167,
        0.604167,
        0.592593,
        0.592593,
        0.763158,
        0.763158,
    ]


def test_custom_channel_mapping(dm_aggregates):
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Channel=pl.when(pl.col.Channel == "SMS")
        .then(pl.lit("MyChannel"))
        .otherwise(pl.col.Channel)
        .cast(pl.Categorical)
    )

    summary_by_channel = dm_aggregates.summary_by_channel(
        custom_channels={"MyChannel": "Web"}
    ).collect()
    assert summary_by_channel.height == 3
    assert summary_by_channel.width == 22
    assert summary_by_channel["ChannelDirection"].to_list() == [
        "Email/Outbound",
        "MyChannel/Outbound",
        "Web/Inbound",
    ]
    assert summary_by_channel["ChannelDirectionGroup"].to_list() == [
        "E-mail/Outbound",
        "Web/Outbound",
        "Web/Inbound",
    ]


def test_aggregate_overall_summary(dm_aggregates):
    overall_summary = dm_aggregates.overall_summary().collect()
    assert overall_summary.height == 1
    assert overall_summary.width == 18
    assert overall_summary["Number of Valid Channels"].item() == 3
    assert overall_summary["Actions"].item() == 35
    assert overall_summary["Treatments"].item() == 0

    # 3 valid channels
    # print(overall_summary)

    assert round(overall_summary["OmniChannel"].item(), 5) == 0.65331

    # Force only one channel to be valid
    dm_aggregates.datamart.model_data = dm_aggregates.datamart.model_data.with_columns(
        Positives=pl.when(pl.col.Channel != "SMS")
        .then(pl.lit(0))
        .otherwise("Positives")
    )

    overall_summary = dm_aggregates.overall_summary().collect()

    # print(overall_summary)

    assert overall_summary["Number of Valid Channels"].item() == 1


def test_aggregate_overall_summary_by_time(dm_aggregates):
    overall_summary = dm_aggregates.overall_summary(by_period="1h").collect()
    assert overall_summary.height == 6
    assert overall_summary.width == 19
    assert overall_summary["Number of Valid Channels"].to_list() == [2, 2, 2, 1, 2, 1]
    assert overall_summary["Actions"].to_list() == [34, 35, 34, 31, 33, 34]
    assert overall_summary["Treatments"].to_list() == [0] * 6
    assert [round(x, 6) for x in overall_summary["OmniChannel"].to_list()] == [
        0.728745,
        0.694444,
        0.728745,
        0.0,
        0.668889,
        0.0,
    ]


def test_summary_by_configuration(dm_aggregates):
    configuration_summary = dm_aggregates.summary_by_configuration().collect()
    assert "AGB" in configuration_summary.columns

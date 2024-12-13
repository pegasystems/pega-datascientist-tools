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
    assert summary_by_channel.shape[0] == 3
    assert summary_by_channel.shape[1] == 23
    assert summary_by_channel["Total Number of Actions"].to_list() == [24, 27, 19]


def test_aggregate_overall_summary(agg):
    overall_summary = agg.overall_summary().collect()
    assert overall_summary.shape[0] == 1
    assert overall_summary.shape[1] == 20
    assert overall_summary["Number of Valid Channels"].item() == 3
    assert overall_summary["Total Number of Treatments"].item() == 0

def test_summary_by_configuration(agg):
    configuration_summary = agg.summary_by_configuration().collect()
    assert "AGB" in configuration_summary.columns
    
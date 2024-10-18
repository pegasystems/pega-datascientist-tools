"""
Testing the functionality of the ADMDatamart Aggregates functions
"""

import os
import pathlib
import sys

import polars as pl
import pytest

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")

from pdstools import ADMDatamart
from pdstools.adm.Aggregates import Aggregates


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

# TODO expand testing, more variations etc

def test_aggregate_summary_by_channel(agg):
    assert agg.summary_by_channel().collect().shape[0] == 3
    assert agg.summary_by_channel().collect().shape[1] == 21



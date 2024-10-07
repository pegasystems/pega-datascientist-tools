"""
Testing the functionality of the ADMDatamart functions
"""

import os
import pathlib
import sys

import polars as pl
import pytest

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import ADMDatamart


@pytest.fixture
def sample():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
    )


def test_cdh_sample_init(sample):
    assert sample


def test_cached_properties(sample: ADMDatamart):
    assert sample.unique_channel_direction == {
        "Email/Outbound",
        "SMS/Outbound",
        "Web/Inbound",
    }

    assert sample.unique_channels == {"Email", "SMS", "Web"}

    assert sample.unique_configurations == {"OmniAdaptiveModel"}

    assert sample.unique_predictor_categories == {"Customer", "IH", "Param", "Primary"}


def test_write_then_load(sample: ADMDatamart):
    modeldata_cache, predictordata_cache = sample.save_data("cache")

    assert ADMDatamart(
        model_df=pl.scan_ipc(modeldata_cache),
        predictor_df=pl.scan_ipc(predictordata_cache),
    )
    os.remove(modeldata_cache)
    os.remove(predictordata_cache)


def test_init_without_model_data(sample: ADMDatamart):
    modeldata_cache, predictordata_cache = sample.save_data("cache2")

    models_only = ADMDatamart(model_df=pl.scan_ipc(modeldata_cache))
    predictors_only = ADMDatamart(predictor_df=pl.scan_ipc(predictordata_cache))
    os.remove(modeldata_cache)
    os.remove(predictordata_cache)

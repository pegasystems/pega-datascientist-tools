"""
Testing the functionality of the ADMDatamart functions
"""

import os
import pathlib

import polars as pl
import pytest
from pdstools import ADMDatamart

basePath = pathlib.Path(__file__).parent.parent.parent


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

    ADMDatamart(model_df=pl.scan_ipc(modeldata_cache))
    ADMDatamart(predictor_df=pl.scan_ipc(predictordata_cache))
    os.remove(modeldata_cache)
    os.remove(predictordata_cache)


def test_active_range_Pega7():
    # In this test data, the datamart is often wrong. This data pre-dates (and inspired) the fixes to the calculations in Pega.
    test_data_mdls = f"{basePath}/data/active_range/dmModels.csv.gz"
    test_data_preds = f"{basePath}/data/active_range/dmPredictors.csv.gz"
    dm = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls), predictor_df=pl.scan_csv(test_data_preds)
    )

    # These tests are identical to the tests in the old R version
    # be careful changing values

    active_ranges = dm.active_ranges("664cc653-279f-54ae-926f-694652d89a54").collect()
    print(active_ranges)
    assert active_ranges["idx_min"].item() == 6
    assert active_ranges["idx_max"].item() == 7
    assert round(active_ranges["AUC_Datamart"].item(), 6) == 0.760333
    assert round(active_ranges["AUC_FullRange"].item(), 6) == 0.760333
    assert round(active_ranges["AUC_ActiveRange"].item(), 6) == 0.5

    active_ranges = dm.active_ranges("4574f1fd-13a7-5703-bf38-9374641f370f").collect()
    print(active_ranges)
    assert active_ranges["idx_min"].item() == 0
    assert active_ranges["idx_max"].item() == 2
    assert round(active_ranges["AUC_Datamart"].item(), 6) == 0.557292
    assert round(active_ranges["AUC_FullRange"].item(), 6) == 0.557292
    assert round(active_ranges["AUC_ActiveRange"].item(), 6) == 0.557292


def test_active_range_newer_single():
    test_data_mdls = f"{basePath}/data/active_range/all_1_mdls.csv"
    test_data_preds = f"{basePath}/data/active_range/all_1_preds.csv"
    dm = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls), predictor_df=pl.scan_csv(test_data_preds)
    )

    ar = dm.active_ranges().collect()
    assert ar["idx_min"].item() == 0
    assert ar["idx_max"].item() == 3
    assert round(ar["AUC_Datamart"].item(), 6) == 0.544661
    assert round(ar["AUC_FullRange"].item(), 6) == 0.544661
    assert round(ar["AUC_ActiveRange"].item(), 6) == 0.534542


def test_active_range_Pega8():
    dm = ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data/active_range/CDHSample-Pega8"
    )

    ar = dm.active_ranges().collect()

    assert ar["nActivePredictors"].to_list() == [19, 17, 11, 3]
    # we're NOT okay here!!
    assert ar["idx_min"].to_list() == [0, 0, 0, 1] # the R version is 3, 1, 1, 2
    assert ar["idx_max"].to_list() == [11, 11, 2, 4] # the R version is 10, 10, 2, 4
    assert [round(x,6) for x in ar["AUC_Datamart"].to_list()] == [0.533401, 0.562353, 0.559777, 0.628571]
    assert [round(x,6) for x in ar["AUC_ActiveRange"].to_list()] == [0.5334013, 0.5623530, 0.5597765, 0.5476054]

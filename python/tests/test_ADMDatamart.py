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

    assert sample.unique_predictor_categories == {"Customer", "IH", "Param"}


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
    assert ar["idx_min"].to_list() == [2, 0, 0, 1]  # the R version is +1 so 3, 1, 1, 2
    assert ar["idx_max"].to_list() == [10, 10, 2, 4]
    assert [round(x, 6) for x in ar["AUC_Datamart"].to_list()] == [
        0.533401,
        0.562353,
        0.559777,
        0.628571,
    ]
    assert [round(x, 7) for x in ar["AUC_ActiveRange"].to_list()] == [
        0.5334013,
        0.5623530,
        0.5597765,
        0.5476054,
    ]
    assert [round(x, 7) for x in ar["AUC_FullRange"].to_list()] == [
        0.5460856,
        0.5652007,
        0.5597765,
        0.5781145,
    ]
    assert [round(x, 6) for x in ar["score_min"].to_list()] == [
        -0.620929,
        -0.600759,
        -0.749120,
        -1.425557,
    ]
    assert [round(x, 6) for x in ar["score_max"].to_list()] == [
        0.468903,
        0.681244,
        1.454179,
        -0.425495,
    ]

def _check_cat(dm, pred_name):
    return (
        dm.predictor_data.filter(PredictorName=pred_name)
        .select(pl.col("PredictorCategory").unique())
        .collect()
        .item()
    )

def test_predictor_categorization_default(sample):
    default_cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    assert default_cats == ["Customer", "IH", "Param"]

    # print(sample.predictor_data.select("PredictorName", "PredictorCategory").unique().sort("PredictorName").collect().to_pandas())

    assert _check_cat(sample, "Customer.HealthMatter") == "Customer"
    assert _check_cat(sample, "IH.SMS.Outbound.Loyal.pxLastOutcomeTime.DaysSince") == "IH"
    assert _check_cat(sample, "Classifier") is None


def test_predictor_categorization_custom_expression(sample):
    categorization = pl.when(
        pl.col("PredictorName").cast(pl.Utf8).str.contains("Score")
    ).then(pl.lit("External Model"))

    sample.apply_predictor_categorization(categorization)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    # print(
    #     sample.predictor_data.select("PredictorName", "PredictorCategory")
    #     .unique()
    #     .sort("PredictorName")
    #     .collect()
    # )
    assert cats == ["Customer", "External Model", "IH", "Param"]
    assert _check_cat(sample, "Customer.RiskScore") == "External Model"

def test_predictor_categorization_dictionary(sample):
    categorization = {"XGBoost Model" : "Score"}

    sample.apply_predictor_categorization(categorization)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    # print(
    #     sample.predictor_data.select("PredictorName", "PredictorCategory")
    #     .unique()
    #     .sort("PredictorName")
    #     .collect()
    # )
    assert cats == ["Customer", "IH", "Param", "XGBoost Model"]
    assert _check_cat(sample, "Customer.CreditScore") == "XGBoost Model"

def test_predictor_categorization_dictionary_regexps(sample):

    # Using a reg exp w/o setting the flag should not match anything
    categorization = {"XGBoost Model" : "Score$"}
    sample.apply_predictor_categorization(categorization)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    assert "XGBoost Model" not in cats

    # But with the flag we should get some more results
    categorization = {"XGBoost Model" : "Score$"}
    sample.apply_predictor_categorization(categorization, use_regexp=True)

    cats = (
        sample.predictor_data.select(pl.col("PredictorCategory").unique())
        .filter(pl.col("PredictorCategory").is_not_null())
        .sort("PredictorCategory")
        .collect()["PredictorCategory"]
        .to_list()
    )
    assert "XGBoost Model" in cats

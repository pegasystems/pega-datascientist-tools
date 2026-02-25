"""Testing the functionality of the ADMDatamart active_ranges function"""

import pathlib

import polars as pl
import pytest
from pdstools import ADMDatamart

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def sample():
    """Fixture to serve as class to call functions from."""
    return ADMDatamart.from_ds_export(
        base_path=f"{basePath}/data/active_range/CDHSample-Pega8",
    )


def test_active_ranges_basic(sample):
    """Test the basic functionality of active_ranges."""
    # Test with no model_id (all models)
    ar = sample.active_ranges().collect()

    # Check that the expected columns are present
    expected_columns = [
        "ModelID",
        "AUC_Datamart",
        "AUC_FullRange",
        "AUC_ActiveRange",
        "Bins",
        "nActivePredictors",
        "classifierLogOffset",
        "sumMinLogOdds",
        "sumMaxLogOdds",
        "score_min",
        "score_max",
        "idx_min",
        "idx_max",
    ]
    for col in expected_columns:
        assert col in ar.columns

    # Check that the number of rows matches the expected number of models
    assert ar.height == 4

    # Check that the values are within expected ranges
    assert all(0 <= auc <= 1 for auc in ar["AUC_Datamart"])
    assert all(0 <= auc <= 1 for auc in ar["AUC_FullRange"])
    assert all(0 <= auc <= 1 for auc in ar["AUC_ActiveRange"])
    assert all(bins > 0 for bins in ar["Bins"])
    assert all(n >= 0 for n in ar["nActivePredictors"])
    assert all(idx_min >= 0 for idx_min in ar["idx_min"])
    assert all(idx_max > 0 for idx_max in ar["idx_max"])
    assert all(
        idx_max >= idx_min for idx_min, idx_max in zip(ar["idx_min"], ar["idx_max"])
    )


def test_active_ranges_single_model(sample):
    """Test active_ranges with a single model ID."""
    # Get all model IDs
    all_models = sample.active_ranges().collect()
    model_id = all_models["ModelID"][0]

    # Test with a single model_id as string
    ar_single = sample.active_ranges(model_id).collect()
    assert ar_single.height == 1
    assert ar_single["ModelID"][0] == model_id

    # Test with a single model_id in a list
    ar_single_list = sample.active_ranges([model_id]).collect()
    assert ar_single_list.height == 1
    assert ar_single_list["ModelID"][0] == model_id

    # Compare results from both methods
    for col in ar_single.columns:
        assert ar_single[col][0] == ar_single_list[col][0]


def test_active_ranges_multiple_models(sample):
    """Test active_ranges with multiple model IDs."""
    # Get all model IDs
    all_models = sample.active_ranges().collect()
    model_ids = all_models["ModelID"][:2].to_list()

    # Test with multiple model_ids
    ar_multiple = sample.active_ranges(model_ids).collect()
    assert ar_multiple.height == 2
    assert set(ar_multiple["ModelID"]) == set(model_ids)


def test_active_ranges_nonexistent_model(sample):
    """Test active_ranges with a nonexistent model ID."""
    # Test with a nonexistent model_id
    ar_nonexistent = sample.active_ranges("nonexistent_model_id").collect()
    assert ar_nonexistent.height == 0


def test_active_ranges_edge_cases():
    """Test active_ranges with edge cases."""
    # Test with empty data
    dm_empty = ADMDatamart(model_df=None, predictor_df=None)
    with pytest.raises(AttributeError):
        dm_empty.active_ranges().collect()

    # Test with model data but no predictor data
    test_data_mdls = f"{basePath}/data/active_range/all_1_mdls.csv"
    dm_no_predictors = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls),
        predictor_df=None,
    )
    with pytest.raises(AttributeError):
        dm_no_predictors.active_ranges().collect()


def test_active_ranges_pega7():
    """Test active_ranges with Pega 7 data."""
    test_data_mdls = f"{basePath}/data/active_range/dmModels.csv.gz"
    test_data_preds = f"{basePath}/data/active_range/dmPredictors.csv.gz"
    dm = ADMDatamart(
        model_df=pl.scan_csv(test_data_mdls),
        predictor_df=pl.scan_csv(test_data_preds),
    )

    # Test specific model with known values
    model_id = "664cc653-279f-54ae-926f-694652d89a54"
    ar = dm.active_ranges(model_id).collect()

    # Check that the values match expected values
    assert ar["idx_min"].item() == 6
    assert ar["idx_max"].item() == 7
    assert round(ar["AUC_Datamart"].item(), 6) == 0.760333
    assert round(ar["AUC_FullRange"].item(), 6) == 0.760333
    assert round(ar["AUC_ActiveRange"].item(), 6) == 0.5

    # Check that AUC_ActiveRange is different from AUC_FullRange when idx_min and idx_max don't span the full range
    assert ar["AUC_ActiveRange"].item() != ar["AUC_FullRange"].item()

    # Test another model where AUC_ActiveRange equals AUC_FullRange
    model_id = "4574f1fd-13a7-5703-bf38-9374641f370f"
    ar = dm.active_ranges(model_id).collect()
    assert round(ar["AUC_ActiveRange"].item(), 6) == round(
        ar["AUC_FullRange"].item(),
        6,
    )

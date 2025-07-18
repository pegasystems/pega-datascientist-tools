"""
Testing the functionality of the ADMDatamart Aggregates predictors_overview function
"""

import pathlib

import pytest
import polars as pl
from pdstools import ADMDatamart
from pdstools.adm.Aggregates import Aggregates

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


def test_predictors_overview(dm_aggregates):
    """Test the predictors_overview method."""
    # Test with no model_id (all models)
    overview = dm_aggregates.predictors_overview().collect()
    assert overview is not None
    assert isinstance(overview, pl.DataFrame)
    
    # Check that the expected columns are present
    expected_columns = [
        "ModelID", "PredictorName", "Responses", "Positives", "EntryType", 
        "isActive", "GroupIndex", "Type", "Univariate Performance", 
        "Bins", "Missing %", "Residual %"
    ]
    for col in expected_columns:
        assert col in overview.columns
    
    # Test with a specific model_id
    model_id = "692f453a-f825-5a90-ab40-f83cbb34b058"
    overview_single = dm_aggregates.predictors_overview(model_id=model_id).collect()
    assert overview_single is not None
    assert isinstance(overview_single, pl.DataFrame)
    assert "ModelID" not in overview_single.columns  # ModelID column should be removed
    assert "PredictorName" in overview_single.columns
    
    # Test with additional aggregations
    additional_aggs = [
        pl.col("BinResponseCount").sum().alias("Total Responses")
    ]
    overview_with_aggs = dm_aggregates.predictors_overview(additional_aggregations=additional_aggs)
    assert "Total Responses" in overview_with_aggs.collect_schema().names()


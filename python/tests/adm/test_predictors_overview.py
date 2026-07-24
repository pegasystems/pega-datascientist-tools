"""Testing the functionality of the ADMDatamart Aggregates predictors_overview function"""

import pathlib

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.adm.Aggregates import Aggregates

basePath = pathlib.Path(__file__).parent.parent.parent.parent


@pytest.fixture
def dm_aggregates():
    """Fixture to serve as class to call functions from."""
    return Aggregates(
        ADMDatamart.from_ds_export(
            base_path=f"{basePath}/data",
            model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
            predictor_filename="Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip",
        ),
    )


def test_predictors_overview(dm_aggregates):
    """Test the predictors_overview method."""
    # Test with no model_id (all models)
    overview = dm_aggregates.predictors_overview().collect()

    expected_columns = [
        "ModelID",
        "PredictorName",
        "PredictorCategory",
        "Responses",
        "Positives",
        "EntryType",
        "isActive",
        "GroupIndex",
        "Type",
        "Univariate Performance",
        "Bins",
        "Missing %",
        "Residual %",
    ]
    assert overview.columns == expected_columns
    assert overview.shape == (1800, 13)

    # Test with a specific model_id
    model_id = "692f453a-f825-5a90-ab40-f83cbb34b058"
    overview_single = dm_aggregates.predictors_overview(model_id=model_id).collect()
    assert overview_single.columns == expected_columns[1:]
    assert overview_single.shape == (90, 12)
    assert overview_single.head(2).select(
        "PredictorName",
        "Responses",
        "Positives",
        "EntryType",
    ).to_dict(as_series=False) == {
        "PredictorName": ["Customer.Prefix", "Classifier"],
        "Responses": [518, 518],
        "Positives": [13, 13],
        "EntryType": ["Active", "Classifier"],
    }

    # Test with additional aggregations
    additional_aggs = [pl.col("BinResponseCount").sum().alias("Total Responses")]
    overview_with_aggs = dm_aggregates.predictors_overview(
        additional_aggregations=additional_aggs,
    )
    assert "Total Responses" in overview_with_aggs.collect_schema().names()

"""Testing the functionality of the ADMDatamart Aggregates predictors_overview function"""

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.adm.Aggregates import Aggregates


@pytest.fixture
def dm_aggregates(sample: ADMDatamart):
    """Fixture to serve as class to call functions from."""
    return Aggregates(sample)


def test_predictors_overview(dm_aggregates):
    """Test predictor overview aggregation on a bounded canonical model slice."""
    model_id = "05085857-c52c-5daa-b51b-0af576a038af"
    datamart = dm_aggregates.datamart
    model_slice = ADMDatamart(
        model_df=datamart.model_data.filter(pl.col("ModelID") == model_id),
        predictor_df=datamart.predictor_data.filter(pl.col("ModelID") == model_id),
    )
    overview = model_slice.aggregates.predictors_overview().collect()

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
    assert overview.shape == (151, 13)

    overview_single = dm_aggregates.predictors_overview(model_id=model_id).collect()
    assert overview_single.columns == expected_columns[1:]
    assert overview_single.shape == (151, 12)
    assert overview_single.head(2).select(
        "PredictorName",
        "Responses",
        "Positives",
        "EntryType",
    ).to_dict(as_series=False) == {
        "PredictorName": ["Classifier", "Customer.AnnualIncome"],
        "Responses": [114460, 114460],
        "Positives": [3916, 3916],
        "EntryType": ["Classifier", "Inactive"],
    }

    # Test with additional aggregations
    additional_aggs = [pl.col("BinResponseCount").sum().alias("Total Responses")]
    overview_with_aggs = dm_aggregates.predictors_overview(
        additional_aggregations=additional_aggs,
    )
    assert "Total Responses" in overview_with_aggs.collect_schema().names()

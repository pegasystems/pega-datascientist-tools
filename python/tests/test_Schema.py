"""
Testing the functionality of the adm Schema module
"""

import polars as pl
from pdstools.adm.Schema import ADMModelSnapshot, ADMPredictorBinningSnapshot


def test_adm_model_snapshot_schema():
    """Test that the ADMModelSnapshot schema is defined with the expected data types"""
    schema = ADMModelSnapshot

    # Check a few key fields
    assert schema.pyModelID == pl.Utf8
    assert schema.pyPerformance == pl.Float64
    assert schema.pySuccessRate == pl.Float64
    assert schema.pyResponseCount == pl.Float32
    assert schema.pyActivePredictors == pl.UInt16
    assert schema.pyTotalPredictors == pl.UInt16

    # Check categorical fields
    categorical_fields = [
        "pxApplication",
        "pyAppliesToClass",
        "pyConfigurationName",
        "pyIssue",
        "pyGroup",
        "pyChannel",
        "pyDirection",
        "pxObjClass",
    ]
    for field in categorical_fields:
        assert getattr(schema, field) == pl.Categorical, (
            f"Field {field} should be Categorical"
        )

    # Check datetime fields
    datetime_fields = [
        "pySnapshotTime",
        "pxSaveDateTime",
        "pxCommitDateTime",
        "pyFactoryUpdatetime",
    ]
    for field in datetime_fields:
        assert getattr(schema, field) == pl.Datetime, (
            f"Field {field} should be Datetime"
        )


def test_adm_predictor_binning_snapshot_schema():
    """Test that the ADMPredictorBinningSnapshot schema is defined with the expected data types"""
    schema = ADMPredictorBinningSnapshot

    # Check a few key fields
    assert schema.pyModelID == pl.Utf8
    assert schema.pyPredictorName == pl.Categorical
    assert schema.pyPerformance == pl.Float64
    assert schema.pyPositives == pl.Float32
    assert schema.pyNegatives == pl.Float32
    assert schema.pyTotalBins == pl.UInt16
    assert schema.pyResponseCount == pl.Float32

    # Check categorical fields
    categorical_fields = [
        "pxObjClass",
        "pyPredictorName",
        "pyType",
        "pyBinType",
        "pyEntryType",
    ]
    for field in categorical_fields:
        assert getattr(schema, field) == pl.Categorical, (
            f"Field {field} should be Categorical"
        )

    # Check datetime fields
    datetime_fields = ["pxCommitDateTime", "pxSaveDateTime", "pySnapshotTime"]
    for field in datetime_fields:
        assert getattr(schema, field) == pl.Datetime, (
            f"Field {field} should be Datetime"
        )


def test_create_dataframe_with_schema():
    """Test creating a DataFrame using the schema definitions"""
    # Create a simple DataFrame with the ADMModelSnapshot schema
    data = {
        "pyModelID": ["model1", "model2"],
        "pyPerformance": [0.8, 0.75],
        "pySuccessRate": [0.6, 0.55],
        "pyResponseCount": [1000, 1200],
        "pyActivePredictors": [10, 12],
        "pyTotalPredictors": [15, 18],
        "pyChannel": ["Web", "Mobile"],
    }

    # Extract schema for these columns
    schema = {col: getattr(ADMModelSnapshot, col) for col in data.keys()}

    # Create DataFrame with schema
    df = pl.DataFrame(data)

    # Check that the schema matches the expected types
    assert df.schema["pyModelID"] == pl.Utf8
    assert df.schema["pyPerformance"] == pl.Float64
    assert df.schema["pySuccessRate"] == pl.Float64
    assert (
        df.schema["pyResponseCount"] == pl.Int64
    )  # Note: This will be Int64 by default, not Float32
    assert (
        df.schema["pyActivePredictors"] == pl.Int64
    )  # Note: This will be Int64 by default, not UInt16
    assert (
        df.schema["pyTotalPredictors"] == pl.Int64
    )  # Note: This will be Int64 by default, not UInt16

    # Create DataFrame with explicit schema
    df_with_schema = pl.DataFrame(data, schema=schema)

    # Check that the schema was applied correctly
    assert df_with_schema.schema["pyModelID"] == pl.Utf8
    assert df_with_schema.schema["pyPerformance"] == pl.Float64
    assert df_with_schema.schema["pySuccessRate"] == pl.Float64
    assert df_with_schema.schema["pyResponseCount"] == pl.Float32
    assert df_with_schema.schema["pyActivePredictors"] == pl.UInt16
    assert df_with_schema.schema["pyTotalPredictors"] == pl.UInt16
    assert df_with_schema.schema["pyChannel"] == pl.Categorical

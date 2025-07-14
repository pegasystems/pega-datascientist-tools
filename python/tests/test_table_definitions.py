"""
Testing the functionality of the table_definitions module
"""

import polars as pl
import pytest
from pdstools.utils.table_definitions import PegaDefaultTables


def test_adm_model_snapshot_schema():
    """Test that the ADMModelSnapshot schema is defined with the expected data types"""
    schema = PegaDefaultTables.ADMModelSnapshot
    
    # Check a few key fields
    assert schema.pyModelID == pl.Utf8
    assert schema.pyPerformance == pl.Float64
    assert schema.pySuccessRate == pl.Float64
    assert schema.pyResponseCount == pl.Float32
    assert schema.pyActivePredictors == pl.UInt16
    assert schema.pyTotalPredictors == pl.UInt16
    
    # Check categorical fields
    categorical_fields = [
        'pxApplication', 'pyAppliesToClass', 'pyConfigurationName',
        'pyIssue', 'pyGroup', 'pyChannel', 'pyDirection', 'pxObjClass'
    ]
    for field in categorical_fields:
        assert getattr(schema, field) == pl.Categorical, f"Field {field} should be Categorical"
    
    # Check datetime fields
    datetime_fields = ['pySnapshotTime', 'pxSaveDateTime', 'pxCommitDateTime', 'pyFactoryUpdatetime']
    for field in datetime_fields:
        assert getattr(schema, field) == pl.Datetime, f"Field {field} should be Datetime"


def test_adm_predictor_binning_snapshot_schema():
    """Test that the ADMPredictorBinningSnapshot schema is defined with the expected data types"""
    schema = PegaDefaultTables.ADMPredictorBinningSnapshot
    
    # Check a few key fields
    assert schema.pyModelID == pl.Utf8
    assert schema.pyPredictorName == pl.Categorical
    assert schema.pyPerformance == pl.Float64
    assert schema.pyPositives == pl.Float32
    assert schema.pyNegatives == pl.Float32
    assert schema.pyTotalBins == pl.UInt16
    assert schema.pyResponseCount == pl.Float32
    
    # Check categorical fields
    categorical_fields = ['pxObjClass', 'pyPredictorName', 'pyType', 'pyBinType', 'pyEntryType']
    for field in categorical_fields:
        assert getattr(schema, field) == pl.Categorical, f"Field {field} should be Categorical"
    
    # Check datetime fields
    datetime_fields = ['pxCommitDateTime', 'pxSaveDateTime', 'pySnapshotTime']
    for field in datetime_fields:
        assert getattr(schema, field) == pl.Datetime, f"Field {field} should be Datetime"


def test_py_value_finder_schema():
    """Test that the pyValueFinder schema is defined with the expected data types"""
    schema = PegaDefaultTables.pyValueFinder
    
    # Check a few key fields
    assert schema.pyDirection == pl.Categorical
    assert schema.pySubjectType == pl.Categorical
    assert schema.ModelPositives == pl.UInt32
    assert schema.pyGroup == pl.Categorical
    assert schema.pyPropensity == pl.Float64
    assert schema.FinalPropensity == pl.Float64
    assert schema.pyStage == pl.Categorical
    assert schema.pxRank == pl.UInt16
    assert schema.pxPriority == pl.Float64
    assert schema.pyModelPropensity == pl.Float64
    assert schema.pyChannel == pl.Categorical
    assert schema.Value == pl.Float64
    assert schema.pyName == pl.Utf8
    assert schema.StartingEvidence == pl.UInt32
    assert schema.pySubjectID == pl.Utf8
    assert schema.DecisionTime == pl.Datetime
    assert schema.pyTreatment == pl.Utf8
    assert schema.pyIssue == pl.Categorical


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
        "pxApplication": ["app1", "app2"]
    }
    
    # Extract schema for these columns
    schema = {
        col: getattr(PegaDefaultTables.ADMModelSnapshot, col)
        for col in data.keys()
    }
    
    # Create DataFrame with schema
    df = pl.DataFrame(data, schema=schema)
    
    # Check that the schema was applied correctly
    assert df.schema["pyModelID"] == pl.Utf8
    assert df.schema["pyPerformance"] == pl.Float64
    assert df.schema["pySuccessRate"] == pl.Float64
    assert df.schema["pyResponseCount"] == pl.Float32
    assert df.schema["pyActivePredictors"] == pl.UInt16
    assert df.schema["pyTotalPredictors"] == pl.UInt16
    assert df.schema["pxApplication"] == pl.Categorical

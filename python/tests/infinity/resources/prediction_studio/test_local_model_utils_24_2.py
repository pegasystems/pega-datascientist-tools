import re

import polars as pl
# Compatibility patches
import onnx
if not hasattr(onnx, 'mapping') and hasattr(onnx, '_mapping'):
    onnx.mapping = onnx._mapping

# Patch for ml_dtypes in Python 3.13+
import ml_dtypes
if not hasattr(ml_dtypes, 'float4_e2m1fn'):
    if hasattr(ml_dtypes, 'float8_e4m3fn'):
        ml_dtypes.float4_e2m1fn = ml_dtypes.float8_e4m3fn
    else:
        # Fallback if neither attribute exists
        # This will at least allow the module to import without error
        import numpy as np
        ml_dtypes.float4_e2m1fn = np.float32
import pytest
from pydantic import ValidationError
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pdstools.infinity.resources.prediction_studio.local_model_utils import (
    Metadata,
    ONNXModel,
    ONNXModelCreationError,
    ONNXModelValidationError,
    OutcomeType,
    Output,
)


def get_regression_pipeline():
    california = fetch_california_housing()
    X, y = california.data, california.target
    pipeline = Pipeline([("regressor", LinearRegression())])
    pipeline.fit(X, y)
    return pipeline


def get_classification_onnx_model():
    iris = load_iris()
    X, y = iris.data, [iris.target_names[i] for i in iris.target]
    cleaned_names = [re.sub(r"\W+", "_", name) for name in iris.feature_names]
    X_df = pl.DataFrame(X, cleaned_names)
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), cleaned_names)]
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("regressor", RandomForestClassifier())]
    )
    pipeline.fit(X_df, y)
    initial_types = [(col, FloatTensorType([None, 1])) for col in cleaned_names]
    metadata = Metadata(
        type=OutcomeType.CATEGORICAL,
        output=Output(
            label_name="output_label",
            score_name="output_probability",
            possible_values=["setosa", "versicolor", "virginica"],
        ),
    )
    return ONNXModel.from_sklearn_pipeline(pipeline, initial_types).add_metadata(
        metadata
    ), X_df.head(1)


def test_onnx_create_from_model_proto():
    onnx_model = get_classification_onnx_model()
    assert ONNXModel.from_onnx_proto(model=onnx_model[0]._model) is not None


def test_validate_and_run_classification_onnx_model():
    onnx_model, X_df = get_classification_onnx_model()
    assert onnx_model.validate() is not None

    df = X_df.with_columns([pl.col(col).cast(pl.Float32) for col in X_df.columns])
    test_data = {col: df[col].to_numpy().reshape(-1, 1) for col in df.columns}
    assert onnx_model.run(test_data)[0][0] == "setosa"


def test_onnx_creation_fails():
    with pytest.raises(ONNXModelCreationError):
        ONNXModel.from_sklearn_pipeline(model=LinearRegression(), initial_types=None)


@pytest.mark.parametrize(
    "model, initial_types, metadata",
    [
        (
            get_regression_pipeline(),
            [("float_input", FloatTensorType([None, 8]))],
            Metadata.from_json("""
                {
                    "predictorList": [
                        {"name": "MedInc", "index": 1, "inputName": "float_input"},
                        {"name": "HouseAge", "index": 2, "inputName": "float_input"},
                        {"name": "AveRooms", "index": 3, "inputName": "float_input"},
                        {"name": "AveBedrms", "index": 4, "inputName": "float_input"},
                        {"name": "Population", "index": 5, "inputName": "float_input"},
                        {"name": "AveOccup", "index": 6, "inputName": "float_input"},
                        {"name": "Latitude", "index": 7, "inputName": "float_input"},
                        {"name": "Longitude", "index": 8, "inputName": "float_input"}
                    ],
                    "type": "continuous",
                    "output": {
                        "labelName": "variable",
                        "minValue": 0.0,
                        "maxValue": 6.0
                    }
                }
            """),
        )
    ],
)
def test_validate_regression_onnx_model(model, initial_types, metadata):
    assert (
        ONNXModel.from_sklearn_pipeline(model=model, initial_types=initial_types)
        .add_metadata(metadata)
        .validate()
        is True
    )


@pytest.mark.parametrize(
    "model, initial_types",
    [(get_regression_pipeline(), [("float_input", FloatTensorType([None, 8]))])],
)
def test_validate_onnx_model_without_metadata(model, initial_types):
    with pytest.raises(ONNXModelValidationError):
        ONNXModel.from_sklearn_pipeline(
            model=model, initial_types=initial_types
        ).validate()


@pytest.mark.parametrize(
    "metadata",
    [
        (  # Invalid model type
            """
                {
                    "type": "Timeseries",
                    "output": {
                        "labelName": "variable",
                        "minValue": 0,
                        "maxValue": 6
                    }
                }
            """
        ),
        (  # Missing model type and output
            """{}"""
        ),
        (  # Missing predictor index
            """
                {
                    "predictorList": [
                        {"name": "MedInc", "index": 1, "inputName": "float_input"},
                        {"name": "HouseAge", "index": 2, "inputName": "float_input"},
                        {"name": "AveRooms", "index": 3, "inputName": "float_input"},
                        {"name": "AveBedrms", "index": 4, "inputName": "float_input"},
                        {"name": "Population", "index": 5, "inputName": "float_input"},
                        {"name": "AveOccup", "index": 6, "inputName": "float_input"},
                        {"name": "Latitude", "index": 7, "inputName": "float_input"},
                        {"name": "Longitude", "inputName": "float_input"}
                    ],
                    "type": "continuous",
                    "output": {
                        "labelName": "variable",
                        "minValue": 0.0,
                        "maxValue": 6.0
                    }
                }
            """
        ),
        (  # Missing predictor name
            """
                {
                    "predictorList": [
                        {"name": "MedInc", "index": 1, "inputName": "float_input"},
                        {"name": "HouseAge", "index": 2, "inputName": "float_input"},
                        {"name": "AveRooms", "index": 3, "inputName": "float_input"},
                        {"name": "AveBedrms", "index": 4, "inputName": "float_input"},
                        {"name": "Population", "index": 5, "inputName": "float_input"},
                        {"name": "AveOccup", "index": 6, "inputName": "float_input"},
                        {"name": "Latitude", "index": 7, "inputName": "float_input"},
                        {"index": 8, "inputName": "float_input"}
                    ],
                    "type": "continuous",
                    "output": {
                        "labelName": "variable",
                        "minValue": 0.0,
                        "maxValue": 6.0
                    }
                }
            """
        ),
        (  # Missing predictor input name
            """
                {
                    "predictorList": [
                        {"name": "MedInc", "index": 1, "inputName": "float_input"},
                        {"name": "HouseAge", "index": 2, "inputName": "float_input"},
                        {"name": "AveRooms", "index": 3, "inputName": "float_input"},
                        {"name": "AveBedrms", "index": 4, "inputName": "float_input"},
                        {"name": "Population", "index": 5, "inputName": "float_input"},
                        {"name": "AveOccup", "index": 6, "inputName": "float_input"},
                        {"name": "Latitude", "index": 7, "inputName": "float_input"},
                        {"name": "Longitude", "index": 8}
                    ],
                    "type": "continuous",
                    "output": {
                        "labelName": "variable",
                        "minValue": 0.0,
                        "maxValue": 6.0
                    }
                }
            """
        ),
    ],
)
def test_onnx_metadata_creation(metadata):
    with pytest.raises(ValidationError):
        Metadata.from_json(metadata)


@pytest.mark.parametrize(
    "model, initial_types, metadata",
    [
        (  # Invalid output label name
            get_regression_pipeline(),
            [("float_input", FloatTensorType([None, 8]))],
            Metadata.from_json("""
                {
                    "predictorList": [
                        {"name": "MedInc", "index": 1, "inputName": "float_input"},
                        {"name": "HouseAge", "index": 2, "inputName": "float_input"},
                        {"name": "AveRooms", "index": 3, "inputName": "float_input"},
                        {"name": "AveBedrms", "index": 4, "inputName": "float_input"},
                        {"name": "Population", "index": 5, "inputName": "float_input"},
                        {"name": "AveOccup", "index": 6, "inputName": "float_input"},
                        {"name": "Latitude", "index": 7, "inputName": "float_input"},
                        {"name": "Longitude", "index": 8, "inputName": "float_input"}
                    ],
                    "type": "continuous",
                    "output": {
                        "labelName": "label",
                        "minValue": 0.0,
                        "maxValue": 6.0
                    }
                }
            """),
        ),
        (  # Duplicate predictor index
            get_regression_pipeline(),
            [("float_input", FloatTensorType([None, 8]))],
            Metadata.from_json("""
                {
                    "predictorList": [
                        {"name": "MedInc", "index": 1, "inputName": "float_input"},
                        {"name": "HouseAge", "index": 2, "inputName": "float_input"},
                        {"name": "AveRooms", "index": 3, "inputName": "float_input"},
                        {"name": "AveBedrms", "index": 4, "inputName": "float_input"},
                        {"name": "Population", "index": 5, "inputName": "float_input"},
                        {"name": "AveOccup", "index": 6, "inputName": "float_input"},
                        {"name": "Latitude", "index": 7, "inputName": "float_input"},
                        {"name": "Longitude", "index": 7, "inputName": "float_input"}
                    ],
                    "type": "continuous",
                    "output": {
                        "labelName": "variable",
                        "minValue": 0.0,
                        "maxValue": 6.0
                    }
                }
            """),
        ),
        (  # Missing predictor mapping
            get_regression_pipeline(),
            [("float_input", FloatTensorType([None, 8]))],
            Metadata.from_json("""
                {
                    "predictorList": [
                        {"name": "MedInc", "index": 1, "inputName": "float_input"},
                        {"name": "HouseAge", "index": 2, "inputName": "float_input"},
                        {"name": "AveRooms", "index": 3, "inputName": "float_input"},
                        {"name": "AveBedrms", "index": 4, "inputName": "float_input"},
                        {"name": "Population", "index": 5, "inputName": "float_input"},
                        {"name": "AveOccup", "index": 6, "inputName": "float_input"},
                        {"name": "Latitude", "index": 7, "inputName": "float_input"}
                    ],
                    "type": "continuous",
                    "output": {
                        "labelName": "variable",
                        "minValue": 0.0,
                        "maxValue": 6.0
                    }
                }
            """),
        ),
    ],
)
def test_validate_onnx_model_with_invalid_metadata(model, initial_types, metadata):
    with pytest.raises(ONNXModelValidationError):
        ONNXModel.from_sklearn_pipeline(
            model=model, initial_types=initial_types
        ).add_metadata(metadata).validate()

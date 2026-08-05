import warnings

# Compatibility patches
import onnx
import polars as pl

if not hasattr(onnx, "mapping") and hasattr(onnx, "_mapping"):
    onnx.mapping = onnx._mapping
import pytest
from pdstools.infinity.resources.prediction_studio.local_model_utils import (
    Metadata,
    ONNXModel,
    ONNXModelCreationError,
    ONNXModelValidationError,
    OutcomeType,
    Output,
)
from pydantic import ValidationError
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REGRESSION_INITIAL_TYPES = [("float_input", FloatTensorType([None, 8]))]


def get_regression_pipeline():
    pipeline = Pipeline([("regressor", LinearRegression())])
    pipeline.fit(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        [1.0, 2.0, 3.0],
    )
    return pipeline


def get_classification_onnx_model():
    cleaned_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_df = pl.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.3, 3.3, 6.0, 2.5],
            [5.8, 2.7, 5.1, 1.9],
        ],
        schema=cleaned_names,
        orient="row",
    )
    y = ["setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica"]
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), cleaned_names)],
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestClassifier(n_estimators=1, random_state=0)),
        ],
    )
    with warnings.catch_warnings():
        # scikit-learn <1.9 probes Polars' deprecated dataframe interchange
        # protocol; scikit-learn 1.9 replaces this path with Narwhals.
        warnings.filterwarnings(
            "ignore",
            message="Support for the dataframe interchange protocol is deprecated since version 1\\.40\\.0",
            category=DeprecationWarning,
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
        metadata,
    ), X_df.head(1)


@pytest.fixture(scope="module")
def regression_pipeline():
    return get_regression_pipeline()


@pytest.fixture(scope="module")
def classification_onnx_model():
    return get_classification_onnx_model()


def test_onnx_create_from_model_proto(classification_onnx_model):
    onnx_model, _ = classification_onnx_model
    recreated = ONNXModel.from_onnx_proto(model=onnx_model._model)
    assert recreated._model.SerializeToString() == onnx_model._model.SerializeToString()


def test_validate_and_run_classification_onnx_model(classification_onnx_model):
    onnx_model, X_df = classification_onnx_model
    assert onnx_model.validate() is True

    df = X_df.with_columns([pl.col(col).cast(pl.Float32) for col in X_df.columns])
    test_data = {col: df[col].to_numpy().reshape(-1, 1) for col in df.columns}
    assert onnx_model.run(test_data)[0][0] == "setosa"


def test_onnx_creation_fails():
    with pytest.raises(ONNXModelCreationError):
        ONNXModel.from_sklearn_pipeline(model=LinearRegression(), initial_types=None)


def test_validate_regression_onnx_model(regression_pipeline):
    metadata = Metadata.from_json("""
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
            """)
    assert (
        ONNXModel.from_sklearn_pipeline(
            model=regression_pipeline,
            initial_types=_REGRESSION_INITIAL_TYPES,
        )
        .add_metadata(metadata)
        .validate()
        is True
    )


def test_validate_onnx_model_without_metadata(regression_pipeline):
    with pytest.raises(ONNXModelValidationError):
        ONNXModel.from_sklearn_pipeline(
            model=regression_pipeline,
            initial_types=_REGRESSION_INITIAL_TYPES,
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
    "metadata",
    [
        pytest.param(
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
            id="invalid-output-label",
        ),
        pytest.param(
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
            id="duplicate-predictor-index",
        ),
        pytest.param(
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
            id="missing-predictor-mapping",
        ),
    ],
)
def test_validate_onnx_model_with_invalid_metadata(regression_pipeline, metadata):
    with pytest.raises(ONNXModelValidationError):
        ONNXModel.from_sklearn_pipeline(
            model=regression_pipeline,
            initial_types=_REGRESSION_INITIAL_TYPES,
        ).add_metadata(metadata).validate()

import datetime

import polars as pl
import pytest
from pdstools.infinity.internal._pagination import PaginatedList
from pdstools.infinity.resources.prediction_studio.v26_1.model import Model
from pdstools.infinity.resources.prediction_studio.v24_2.model import Model as ModelV24_2
from pdstools.infinity.resources.prediction_studio.v26_1.prediction import Prediction
from pdstools.infinity.resources.prediction_studio.v24_2.prediction import Prediction as PredictionV24_2
from pdstools.infinity.resources.prediction_studio.v26_1.prediction_studio import (
    PredictionStudio,
)


@pytest.fixture
def prediction_studio_client(mocker):
    client = mocker.MagicMock()
    return PredictionStudio(client=client)


mock_response_repository = {
    "repositoryName": "TestRepo",
    "repositoryType": "S3",
    "bucketName": "test-bucket",
    "rootPath": "/test-path",
    "datamartExportLocation": "/datamart-export",
}

mock_response_model = {
    "models": [
        {
            "modelId": "@BASECLASS!TESTMODEL_FALCONS",
            "label": "testModel_falcons",
            "modelType": "Adaptive model",
            "modelingTechnique": "Adaptive model - Bayesian",
            "source": "Pega",
            "status": "Completed",
            "lastUpdateTime": "20240718T120552.671 GMT",
            "updatedBy": "Somnath Paul",
        },
        {
            "modelId": "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371",
            "label": "Accept",
            "modelType": "Adaptive model",
            "modelingTechnique": "Adaptive model - Bayesian",
            "source": "Pega",
            "status": "Completed",
            "lastUpdateTime": "20240718T104417.891 GMT",
            "updatedBy": "Somnath Paul",
        },
    ],
}

mock_response_predictions = {
    "predictions": [
        {
            "predictionId": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
            "label": "Predict Cards Acceptance",
            "objective": "Accept",
            "subject": "Customer",
            "status": "Completed",
            "lastUpdateTime": "20240718T120557.925 GMT",
        },
        {
            "predictionId": "DATA-DECISION-REQUEST-CUSTOMER!PREDICTACTIONPROPENSITY",
            "label": "Predict Action Propensity",
            "objective": "Propensity to Accept",
            "status": "Completed",
            "lastUpdateTime": "20231129T113556.481 GMT",
        },
    ],
}

mock_response_notifications = {
    "notifications": [
        {
            "description": "Model performance is at its minimum of 50 AUC",
            "modelID": "CDHSAMPLE-DATA-CUSTOMER!ADM_16132577401",
            "modelType": "Adaptive model",
            "notificationType": "Performance",
            "notificationID": "N-3",
            "predictionID": "CDHSAMPLE-DATA-CUSTOMER!CONVERSION",
            "notificationMnemonic": "Model performance is at its minimum",
            "context": "Sales-DepositAccounts-BasicChecking",
            "impact": "High",
            "triggerTime": "20240720T050556.910 GMT",
        },
        {
            "description": "Model performance is at its minimum of 50 AUC",
            "modelID": "DATA-DECISION-REQUEST-CUSTOMER!OMNIADAPTIVEMODEL",
            "modelType": "Adaptive model",
            "notificationType": "Performance",
            "notificationID": "N-3",
            "notificationMnemonic": "Model performance is at its minimum",
            "context": "Sales-DepositAccounts-RegularSaving",
            "predictionID": "DATA-DECISION-REQUEST-CUSTOMER!PREDICTACTIONPROPENSITY",
            "impact": "High",
            "triggerTime": "20240720T050556.826 GMT",
        },
    ],
}


def test_version(mocker):
    """v26 classes report the correct version string."""
    client = mocker.MagicMock()
    ps = PredictionStudio(client=client)
    assert ps.version == "26.1"


def test_repository(prediction_studio_client, mocker):
    mock_get = mocker.patch.object(
        prediction_studio_client._client,
        "get",
        return_value=mock_response_repository,
    )
    result = prediction_studio_client.repository()

    mock_get.assert_called_once_with(
        "/prweb/api/PredictionStudio/v3/predictions/repository",
    )

    assert result.name == "TestRepo"
    assert result.type == "S3"
    assert result.bucket_name == "test-bucket"
    assert result.root_path == "/test-path"
    assert result.datamart_export_location == "/datamart-export"


@pytest.mark.parametrize(
    "return_df, mock_response, expected_type, expected_length, expected_columns",
    [
        (False, mock_response_model, PaginatedList, None, None),
        (
            True,
            mock_response_model,
            pl.DataFrame,
            2,
            [
                "model_id",
                "label",
                "model_type",
                "modeling_technique",
                "source",
                "status",
                "last_update_time",
                "updated_by",
                "performance",
                "performance_measure",
            ],
        ),
    ],
)
def test_list_models(
    prediction_studio_client,
    mocker,
    return_df,
    mock_response,
    expected_type,
    expected_length,
    expected_columns,
):
    method_to_patch = "get" if not return_df else "request"
    mocker.patch.object(
        prediction_studio_client._client,
        method_to_patch,
        return_value=mock_response,
    )
    result = prediction_studio_client.list_models(return_df=return_df)

    assert isinstance(result, expected_type)

    if return_df:
        assert len(result) == expected_length
        assert result.columns == expected_columns
        assert result.shape == (2, 10)
        assert result[0]["model_id"][0] == "@BASECLASS!TESTMODEL_FALCONS"
        assert result[0]["label"][0] == "testModel_falcons"
        assert result[0]["last_update_time"][0] == datetime.datetime(
            2024, 7, 18, 12, 5, 52, 671000,
        )


@pytest.mark.parametrize(
    "fetch_type, fetch_value",
    [
        ("id", "@BASECLASS!TESTMODEL_FALCONS"),
        ("label", "testModel_falcons"),
    ],
)
def test_get_model(prediction_studio_client, mocker, fetch_type, fetch_value):
    mocker.patch.object(
        prediction_studio_client._client,
        "request",
        return_value=mock_response_model,
    )
    if fetch_type == "id":
        result = prediction_studio_client.get_model(model_id=fetch_value)
    elif fetch_type == "label":
        result = prediction_studio_client.get_model(label=fetch_value)

    assert isinstance(result, (Model, ModelV24_2))


@pytest.mark.parametrize(
    "return_df, expected_type, expected_length, expected_columns",
    [
        (False, PaginatedList, None, None),
        (
            True,
            pl.DataFrame,
            2,
            [
                "prediction_id",
                "label",
                "objective",
                "subject",
                "status",
                "last_update_time",
                "type",
                "performance",
                "performance_measure",
            ],
        ),
    ],
)
def test_list_predictions(
    prediction_studio_client,
    mocker,
    return_df,
    expected_type,
    expected_length,
    expected_columns,
):
    method_to_patch = "get" if not return_df else "request"
    mocker.patch.object(
        prediction_studio_client._client,
        method_to_patch,
        return_value=mock_response_predictions,
    )
    result = prediction_studio_client.list_predictions(return_df=return_df)

    assert isinstance(result, expected_type)

    if return_df:
        assert len(result) == expected_length
        assert result.columns == expected_columns


@pytest.mark.parametrize(
    "identifier_type, identifier_value",
    [
        ("id", "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"),
        ("label", "Predict Cards Acceptance"),
    ],
)
def test_get_predictions(
    prediction_studio_client,
    mocker,
    identifier_type,
    identifier_value,
):
    mocker.patch.object(
        prediction_studio_client._client,
        "request",
        return_value=mock_response_predictions,
    )
    if identifier_type == "id":
        result = prediction_studio_client.get_prediction(prediction_id=identifier_value)
    else:
        result = prediction_studio_client.get_prediction(label=identifier_value)

    assert isinstance(result, (Prediction, PredictionV24_2))


def test_trigger_datamart(prediction_studio_client, mocker):
    mock_response = {
        "referenceId": "W-4002",
        "location": "/AWSFalcons/datamart/",
        "repositoryName": "AWSFalcons",
    }

    mock_post = mocker.patch.object(
        prediction_studio_client._client,
        "post",
        return_value=mock_response,
    )
    result = prediction_studio_client.trigger_datamart_export()

    mock_post.assert_called_once_with("/prweb/api/PredictionStudio/v1/datamart/export")

    assert result.reference_id == "W-4002"
    assert result.location == "/AWSFalcons/datamart/"
    assert result.repository_name == "AWSFalcons"


def test_model_categories(prediction_studio_client, mocker):
    mock_response = {
        "categories": [
            {"categoryLabel": "Retention", "categoryName": "Retention"},
            {"categoryLabel": "Acquisition", "categoryName": "Acquisition"},
        ],
    }

    mock_get = mocker.patch.object(
        prediction_studio_client._client,
        "get",
        return_value=mock_response,
    )
    result = prediction_studio_client.get_model_categories()

    mock_get.assert_called_once_with(
        "/prweb/api/PredictionStudio/v3/predictions/modelCategories",
    )

    assert result[0] == {"categoryLabel": "Retention", "categoryName": "Retention"}
    assert result[1] == {"categoryLabel": "Acquisition", "categoryName": "Acquisition"}


@pytest.mark.parametrize(
    "return_df, expected_type, additional_checks",
    [
        (False, PaginatedList, None),
        (
            True,
            pl.DataFrame,
            lambda df: (
                len(df) == 2,
                df.columns
                == [
                    "model_id",
                    "prediction_id",
                    "context",
                    "notification_type",
                    "notification_id",
                    "notification_mnemonic",
                    "description",
                    "model_type",
                    "impact",
                    "trigger_time",
                ],
                df.shape == (2, 10),
            ),
        ),
    ],
)
def test_get_notifications(
    prediction_studio_client,
    mocker,
    return_df,
    expected_type,
    additional_checks,
):
    method_to_patch = "get" if not return_df else "request"
    mocker.patch.object(
        prediction_studio_client._client,
        method_to_patch,
        return_value=mock_response_notifications,
    )
    result = prediction_studio_client.get_notifications(return_df=return_df)

    assert isinstance(result, expected_type)
    if additional_checks is not None:
        checks = additional_checks(result)
        assert all(checks)


def test_get_reports(prediction_studio_client, mocker):
    mock_response = {
        "reports": [
            {"reportId": "R-1", "name": "Weekly Performance"},
            {"reportId": "R-2", "name": "Monthly Summary"},
        ],
    }

    mock_get = mocker.patch.object(
        prediction_studio_client._client,
        "get",
        return_value=mock_response,
    )
    result = prediction_studio_client.get_reports()

    mock_get.assert_called_once_with(
        "/prweb/api/PredictionStudio/v1/reports",
    )

    assert len(result) == 2
    assert result[0] == {"reportId": "R-1", "name": "Weekly Performance"}
    assert result[1] == {"reportId": "R-2", "name": "Monthly Summary"}


def test_get_settings(prediction_studio_client, mocker):
    mock_response = {
        "repositoryType": "S3",
        "enableMonitoring": True,
    }

    mock_get = mocker.patch.object(
        prediction_studio_client._client,
        "get",
        return_value=mock_response,
    )
    result = prediction_studio_client.get_settings()

    mock_get.assert_called_once_with(
        "/prweb/api/PredictionStudio/v1/settings",
    )

    assert result["repositoryType"] == "S3"
    assert result["enableMonitoring"] is True


def test_list_models_endpoint(prediction_studio_client, mocker):
    """Verify list_models calls the v26 endpoint (v2)."""
    mock_request = mocker.patch.object(
        prediction_studio_client._client,
        "request",
        return_value=mock_response_model,
    )
    prediction_studio_client.list_models(return_df=True)

    mock_request.assert_called_once()
    call_args = mock_request.call_args
    assert call_args[0][1] == "/prweb/api/PredictionStudio/v2/models"


def test_list_predictions_endpoint(prediction_studio_client, mocker):
    """Verify list_predictions calls the v26 endpoint (v3)."""
    mock_request = mocker.patch.object(
        prediction_studio_client._client,
        "request",
        return_value=mock_response_predictions,
    )
    prediction_studio_client.list_predictions(return_df=True)

    mock_request.assert_called_once()
    call_args = mock_request.call_args
    assert call_args[0][1] == "/prweb/api/PredictionStudio/v3/predictions"

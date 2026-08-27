"""Async coverage for the v27_1 Prediction Studio resources.

The async classes are thin coroutine counterparts of the sync ones, but they
carry their own endpoint literals, so this module asserts the v5 paths
independently rather than trusting the sync tests to cover them.
"""

from unittest.mock import AsyncMock

import polars as pl
import pytest
from pdstools.infinity.resources.prediction_studio.v27_1.prediction import (
    AsyncPrediction,
)
from pdstools.infinity.resources.prediction_studio.v27_1.prediction_studio import (
    AsyncPredictionStudio,
)

from .test_prediction import mock_performance_metric, mock_prediction_describe
from .test_prediction_studio import (
    mock_response_model,
    mock_response_notifications,
    mock_response_predictions,
)

PREDICTION_ID = "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"


@pytest.fixture
def client():
    client = AsyncMock()
    client.get = AsyncMock(return_value={})
    client.post = AsyncMock(return_value={})
    client.request = AsyncMock(return_value={})
    return client


@pytest.fixture
def prediction_studio(client):
    return AsyncPredictionStudio(client=client)


@pytest.fixture
def prediction(client):
    return AsyncPrediction(client=client, predictionId=PREDICTION_ID, label="Predict Cards Acceptance")


async def test_repository_reads_v5_settings(prediction_studio, client):
    client.get.return_value = {
        "settings": {
            "generalSettings": {
                "storage": {"key": "Analytics repository", "value": "TestRepo"},
            },
        },
    }

    repository = await prediction_studio.repository()

    client.get.assert_awaited_once_with("/prweb/api/PredictionStudio/v5/settings")
    assert repository.name == "TestRepo"
    # The v5 settings payload carries no repository metadata beyond the name.
    assert repository.type is None
    assert repository.bucket_name is None
    assert repository.root_path is None


async def test_repository_without_storage(prediction_studio, client):
    client.get.return_value = {"settings": {"generalSettings": {}}}

    repository = await prediction_studio.repository()

    assert repository.name is None


async def test_model_categories_read_v5_settings(prediction_studio, client):
    client.get.return_value = {
        "settings": {
            "generalSettings": {
                "modelCategories": [
                    {"category": "Retention", "label": "Retention"},
                    {"category": "Sales", "label": "Sales"},
                ],
            },
        },
    }

    categories = await prediction_studio.get_model_categories()

    client.get.assert_awaited_once_with("/prweb/api/PredictionStudio/v5/settings")
    assert [c["category"] for c in categories] == ["Retention", "Sales"]


async def test_model_categories_without_categories(prediction_studio, client):
    client.get.return_value = {"settings": {"generalSettings": {}}}

    assert await prediction_studio.get_model_categories() == []


async def test_get_settings_uses_v5(prediction_studio, client):
    client.get.return_value = {"settings": {"generalSettings": {}}}

    await prediction_studio.get_settings()

    client.get.assert_awaited_once_with("/prweb/api/PredictionStudio/v5/settings")


async def test_get_reports_uses_v5(prediction_studio, client):
    client.get.return_value = {"reports": [{"name": "report"}]}

    reports = await prediction_studio.get_reports()

    client.get.assert_awaited_once_with("/prweb/api/PredictionStudio/v5/reports")
    assert reports == [{"name": "report"}]


async def test_models_property_targets_v5(prediction_studio):
    assert prediction_studio.models._url == "/prweb/api/PredictionStudio/v5/models"


async def test_predictions_property_targets_v5(prediction_studio):
    assert (
        prediction_studio.predictions._url
        == "/prweb/api/PredictionStudio/v5/predictions"
    )


async def test_trigger_datamart_export_uses_v5(prediction_studio, client):
    client.post.return_value = {
        "referenceId": "REF-1",
        "location": "loc",
        "repositoryName": "TestRepo",
    }

    export = await prediction_studio.trigger_datamart_export()

    client.post.assert_awaited_once_with("/prweb/api/PredictionStudio/v5/datamart/export")
    assert export.reference_id == "REF-1"


async def test_prediction_describe_uses_v5(prediction, client):
    client.get.return_value = {"predictionId": PREDICTION_ID, "label": "Predict"}

    await prediction.describe()

    client.get.assert_awaited_once_with(
        f"/prweb/api/PredictionStudio/v5/predictions/{PREDICTION_ID}",
    )


async def test_prediction_get_staged_changes_uses_v5(prediction, client):
    client.get.return_value = {"listOfChanges": []}

    await prediction.get_staged_changes()

    client.get.assert_awaited_once_with(
        f"/prweb/api/PredictionStudio/v5/predictions/{PREDICTION_ID}/staged",
        data=None,
    )


async def test_prediction_get_champion_challengers_uses_v5(prediction, client):
    client.get.return_value = mock_prediction_describe

    ccs = await prediction.get_champion_challengers()

    client.get.assert_awaited_once_with(
        f"/prweb/api/PredictionStudio/v5/predictions/{PREDICTION_ID}",
    )
    assert len(ccs) == 3
    assert ccs[0].prediction_id == PREDICTION_ID


async def test_prediction_get_metric_uses_v5(prediction, client):
    from datetime import date

    client.get.return_value = mock_performance_metric

    result = await prediction.get_metric(
        start_date=date(2024, 7, 2),
        end_date=date(2024, 7, 11),
        metric="Performance",
        frequency="Daily",
    )

    client.get.assert_awaited_once_with(
        f"/prweb/api/PredictionStudio/v5/predictions/{PREDICTION_ID}/metric/Performance",
        startDate="02/07/2024",
        endDate="11/07/2024",
        frequency="Daily",
    )
    assert result.shape == (8, 3)


async def test_prediction_package_staged_changes_uses_v5(prediction, client):
    client.post.return_value = {"referenceID": "M-3001"}

    result = await prediction.package_staged_changes()

    client.post.assert_awaited_once_with(
        f"/prweb/api/PredictionStudio/v5/predictions/{PREDICTION_ID}/staged",
        data={"reviewNote": "Approving the changes"},
    )
    assert result["referenceID"] == "M-3001"


async def test_prediction_get_notifications_uses_v5(prediction):
    notifications = await prediction.get_notifications()

    assert notifications._url == (
        f"/prweb/api/PredictionStudio/v5/predictions/{PREDICTION_ID}/notifications?category=All"
    )


async def test_prediction_get_notifications_as_df(prediction, client):
    client.request.return_value = mock_response_notifications

    result = await prediction.get_notifications(return_df=True)

    assert isinstance(result, pl.DataFrame)


async def test_prediction_add_conditional_model_uses_v5(prediction, client):
    client.post.return_value = {"referenceID": "M-6011"}
    client.get.return_value = mock_prediction_describe

    cc = await prediction.add_conditional_model(
        new_model="@baseclass!testModel_falcons",
        category="Retention",
    )

    endpoint = client.post.await_args.args[0]
    assert endpoint.startswith("/prweb/api/PredictionStudio/v5/predictions/")
    assert "/category/Retention/models/" in endpoint
    assert cc.prediction_id == PREDICTION_ID


async def test_studio_get_model_uses_v5(prediction_studio, client):
    client.request.return_value = mock_response_model

    model = await prediction_studio.get_model("@BASECLASS!TESTMODEL_FALCONS")

    client.request.assert_awaited_once_with("get", "/prweb/api/PredictionStudio/v5/models", pageSize=100)
    assert model.model_id == "@BASECLASS!TESTMODEL_FALCONS"


async def test_studio_get_prediction_uses_v5(prediction_studio, client):
    client.request.return_value = mock_response_predictions

    result = await prediction_studio.get_prediction(PREDICTION_ID)

    client.request.assert_awaited_once_with("get", "/prweb/api/PredictionStudio/v5/predictions", pageSize=100)
    assert result.prediction_id == PREDICTION_ID


async def test_studio_list_models_uses_v5(prediction_studio, client):
    client.request.return_value = mock_response_model

    result = await prediction_studio.list_models(return_df=True)

    client.request.assert_awaited_once_with("get", "/prweb/api/PredictionStudio/v5/models", pageSize=100)
    assert isinstance(result, pl.DataFrame)


async def test_studio_list_predictions_uses_v5(prediction_studio, client):
    client.request.return_value = mock_response_predictions

    result = await prediction_studio.list_predictions(return_df=True)

    client.request.assert_awaited_once_with("get", "/prweb/api/PredictionStudio/v5/predictions", pageSize=100)
    assert isinstance(result, pl.DataFrame)


async def test_studio_get_notifications_uses_v5(prediction_studio, client):
    client.request.return_value = mock_response_notifications

    result = await prediction_studio.get_notifications(return_df=True)

    assert isinstance(result, pl.DataFrame)

"""Tests for async resource classes: AsyncPredictionStudio, AsyncPrediction, AsyncModel.

These mirror the sync tests in python/tests/prediction_studio/ but exercise
the async code paths, verifying that the write-once mixin pattern works
correctly for the async concrete classes.
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest
from pdstools.infinity.internal._pagination import AsyncPaginatedList
from pdstools.infinity.resources.prediction_studio.v24_2.model import AsyncModel
from pdstools.infinity.resources.prediction_studio.v24_2.prediction import (
    AsyncPrediction,
)
from pdstools.infinity.resources.prediction_studio.v24_2.prediction_studio import (
    AsyncPredictionStudio,
)

# ---------------------------------------------------------------------------
# Helpers — build a mock async client
# ---------------------------------------------------------------------------


def _make_async_client():
    """Create a MagicMock that behaves like AsyncAPIClient."""
    client = MagicMock()
    # Async HTTP methods must be AsyncMock so `await` works.
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.patch = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    client.request = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Mock data (same payloads as the sync tests)
# ---------------------------------------------------------------------------

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
            "description": "Model performance is at its minimum of 50 AUC for model context Sales-DepositAccounts-BasicChecking",
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
            "description": "Model performance is at its minimum of 50 AUC for model context Sales-DepositAccounts-RegularSaving",
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

mock_model_categories = {
    "categories": [
        {"categoryLabel": "Retention", "categoryName": "Retention"},
        {"categoryLabel": "Acquisition", "categoryName": "Acquisition"},
    ],
}

mock_prediction_describe = {
    "predictionId": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
    "label": "Predict Cards Acceptance",
    "objective": "Accept",
    "subject": "Customer",
    "outcomeType": "Binary",
    "responseTimeoutType": "Indefinitely",
    "responseTimeoutValue": "-999",
    "responseTimeoutUnit": "--",
    "delayedlearningtype": "MakeDecision",
    "targetLabels": [{"label": "Accept"}],
    "alternateLabels": [{"label": "Reject"}],
    "context": [
        {
            "contextName": "NoContext",
            "defaultModels": [
                {
                    "id": "CDHSample-Data-Customer!Adm_16330376371",
                    "label": "Accept",
                    "role": "ACTIVE",
                    "type": "Adaptive model",
                    "performance": 72.51,
                    "performanceMeasure": "AUC",
                    "componentName": "Accept",
                    "modelingTechnique": "Adaptive model - Bayesian",
                },
            ],
            "categoryModels": [
                {
                    "id": "@baseclass!testModel_falcons",
                    "label": "testModel_falcons",
                    "role": "ACTIVE",
                    "type": "Adaptive model",
                    "performance": 0.0,
                    "categoryName": "Retention",
                    "componentName": "testModel_falcons",
                    "modelingTechnique": "Adaptive model - Bayesian",
                },
            ],
        },
    ],
    "supportingModels": [
        {
            "id": "CDHSample-Data-Customer!RiskModel",
            "label": "Credit Risk Model",
            "role": "ACTIVE",
            "type": "Predictive model",
            "performance": 50.61,
            "performanceMeasure": "AUC",
            "componentName": "RiskModel",
            "contextName": "NoContext",
            "modelingTechnique": "GBM and XGBoost",
        },
    ],
    "metrics": {"lift": 0.0, "performance": 68.87, "performanceMeasure": "AUC"},
}

mock_model_describe = {
    "modelId": "@BASECLASS!TESTMODEL_FALCONS",
    "status": "Completed",
    "label": "testModel_falcons",
    "type": "Adaptive Model",
    "subject": "@baseclass",
    "outcomeType": "Scoring",
    "modelingTechnique": "Adaptive model",
    "modelCategory": "Retention",
    "source": "Pega",
    "targetLabels": [{"label": "Accept"}],
    "alternateLabels": [{"label": "Reject"}],
    "predictors": [{"name": ".Salary", "dataType": "symbolic"}],
    "parameterizedPredictors": [],
    "IHSummaryPredictors": [
        {
            "name": "IH.{Channel}.{Direction}.{Outcome}.pxLastGroupID",
            "aggregation": "last",
        },
    ],
    "metrics": {"performance": 0.0, "performanceMeasure": "AUC"},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def async_client():
    return _make_async_client()


@pytest.fixture
def async_ps(async_client):
    return AsyncPredictionStudio(client=async_client)


@pytest.fixture
def async_prediction(async_client):
    return AsyncPrediction(
        client=async_client,
        predictionId="CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
        label="Predict Cards Acceptance",
        objective="Accept",
        subject="Customer",
        status="Completed",
        lastUpdateTime="20240718T120557.925 GMT",
    )


@pytest.fixture
def async_model(async_client):
    return AsyncModel(
        client=async_client,
        modelId="@BASECLASS!TESTMODEL_FALCONS",
        label="testModel_falcons",
        modelType="Adaptive model",
        modelingTechnique="Adaptive model - Bayesian",
        source="Pega",
        status="Completed",
        lastUpdateTime="20240718T120552.671 GMT",
        updatedBy="Somnath Paul",
    )


# ---------------------------------------------------------------------------
# AsyncPredictionStudio
# ---------------------------------------------------------------------------


class TestAsyncPredictionStudio:
    @pytest.mark.asyncio
    async def test_repository(self, async_ps, async_client):
        async_client.get.return_value = mock_response_repository
        repo = await async_ps.repository()

        async_client.get.assert_awaited_once_with(
            "/prweb/api/PredictionStudio/v3/predictions/repository",
        )
        assert repo.name == "TestRepo"
        assert repo.type == "S3"
        assert repo.bucket_name == "test-bucket"
        assert repo.root_path == "/test-path"
        assert repo.datamart_export_location == "/datamart-export"

    @pytest.mark.asyncio
    async def test_list_models_returns_paginated_list(self, async_ps, async_client):
        async_client.get.return_value = mock_response_model
        result = await async_ps.list_models(return_df=False)
        assert isinstance(result, AsyncPaginatedList)

    @pytest.mark.asyncio
    async def test_list_models_as_df(self, async_ps, async_client):
        async_client.get.return_value = mock_response_model
        async_client.request.return_value = mock_response_model
        result = await async_ps.list_models(return_df=True)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 8)
        assert result.columns == [
            "model_id",
            "label",
            "model_type",
            "modeling_technique",
            "source",
            "status",
            "last_update_time",
            "updated_by",
        ]

    @pytest.mark.asyncio
    async def test_list_predictions_returns_paginated_list(
        self,
        async_ps,
        async_client,
    ):
        async_client.get.return_value = mock_response_predictions
        result = await async_ps.list_predictions(return_df=False)
        assert isinstance(result, AsyncPaginatedList)

    @pytest.mark.asyncio
    async def test_list_predictions_as_df(self, async_ps, async_client):
        async_client.get.return_value = mock_response_predictions
        async_client.request.return_value = mock_response_predictions
        result = await async_ps.list_predictions(return_df=True)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 6)
        assert result.columns == [
            "prediction_id",
            "label",
            "objective",
            "subject",
            "status",
            "last_update_time",
        ]

    @pytest.mark.asyncio
    async def test_get_model_categories(self, async_ps, async_client):
        """get_model_categories is an @api_method on the mixin — tests async path."""
        async_client.get.return_value = mock_model_categories
        result = await async_ps.get_model_categories()

        async_client.get.assert_awaited_once_with(
            "/prweb/api/PredictionStudio/v3/predictions/modelCategories",
        )
        assert len(result) == 2
        assert result[0] == {"categoryLabel": "Retention", "categoryName": "Retention"}

    @pytest.mark.asyncio
    async def test_get_notifications_returns_paginated_list(
        self,
        async_ps,
        async_client,
    ):
        async_client.get.return_value = mock_response_notifications
        result = await async_ps.get_notifications(return_df=False)
        assert isinstance(result, AsyncPaginatedList)

    @pytest.mark.asyncio
    async def test_get_notifications_as_df(self, async_ps, async_client):
        async_client.get.return_value = mock_response_notifications
        async_client.request.return_value = mock_response_notifications
        result = await async_ps.get_notifications(return_df=True)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 10)

    @pytest.mark.asyncio
    async def test_trigger_datamart_export(self, async_ps, async_client):
        mock_export = {
            "referenceId": "W-4002",
            "location": "/AWSFalcons/datamart/",
            "repositoryName": "AWSFalcons",
        }
        async_client.post.return_value = mock_export
        result = await async_ps.trigger_datamart_export()

        async_client.post.assert_awaited_once_with(
            "/prweb/api/PredictionStudio/v1/datamart/export",
        )
        assert result.reference_id == "W-4002"
        assert result.location == "/AWSFalcons/datamart/"
        assert result.repository_name == "AWSFalcons"

    @pytest.mark.asyncio
    async def test_version_attribute(self, async_ps):
        assert async_ps.version == "24.2"

    @pytest.mark.asyncio
    async def test_upload_model_file_path(self, async_ps, async_client, tmp_path):
        """upload_model should base64-encode a file-based LocalModel and POST it."""
        from pdstools.infinity.resources.prediction_studio.local_model_utils import (
            PMMLModel,
        )

        model_path = tmp_path / "test.model"
        model_path.write_bytes(b"hello model bytes")

        async_client.post.return_value = {
            "repositoryName": "AWSFalcons",
            "filePath": "model-staging/test.model",
        }

        result = await async_ps.upload_model(
            PMMLModel(str(model_path)),
            file_name="test.model",
        )

        # Verify the correct endpoint + base64-encoded payload
        import base64

        expected_b64 = base64.encodebytes(b"hello model bytes").decode("ascii")
        async_client.post.assert_awaited_once_with(
            "/prweb/api/PredictionStudio/v1/model",
            data={"fileSource": expected_b64, "fileName": "test.model"},
        )

        assert result.repository_name == "AWSFalcons"
        assert result.file_path == "model-staging/test.model"

    @pytest.mark.asyncio
    async def test_upload_model_onnx(self, async_ps, async_client, monkeypatch):
        """ONNX branch serialises via SerializeToString + base64."""
        import base64
        from unittest.mock import MagicMock

        from pdstools.infinity.resources.prediction_studio.local_model_utils import (
            ONNXModel,
        )

        # Build an ONNXModel without running the real constructor (which requires onnx).
        model = ONNXModel.__new__(ONNXModel)
        from pdstools.infinity.resources.prediction_studio.base import LocalModel

        LocalModel.__init__(model)
        inner = MagicMock()
        inner.SerializeToString.return_value = b"\x08\x01onnx-bytes"
        model._model = inner
        # Skip the real onnx validation
        monkeypatch.setattr(ONNXModel, "validate", lambda self: True)

        async_client.post.return_value = {
            "repositoryName": "Repo",
            "filePath": "path/model.onnx",
        }
        result = await async_ps.upload_model(model, file_name="model.onnx")

        expected_b64 = base64.encodebytes(b"\x08\x01onnx-bytes").decode("ascii")
        async_client.post.assert_awaited_once_with(
            "/prweb/api/PredictionStudio/v1/model",
            data={"fileSource": expected_b64, "fileName": "model.onnx"},
        )
        assert result.repository_name == "Repo"
        assert result.file_path == "path/model.onnx"


# ---------------------------------------------------------------------------
# AsyncPrediction
# ---------------------------------------------------------------------------


class TestAsyncPrediction:
    @pytest.mark.asyncio
    async def test_init_attributes(self, async_prediction):
        assert async_prediction.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
        assert async_prediction.label == "Predict Cards Acceptance"
        assert async_prediction.objective == "Accept"
        assert async_prediction.subject == "Customer"
        assert async_prediction.status == "Completed"
        assert isinstance(async_prediction.last_update_time, datetime.datetime)

    @pytest.mark.asyncio
    async def test_describe(self, async_prediction, async_client):
        async_client.get.return_value = mock_prediction_describe
        result = await async_prediction.describe()

        async_client.get.assert_awaited_once_with(
            "/prweb/api/PredictionStudio/v3/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
        )
        assert result["predictionId"] == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
        assert result["label"] == "Predict Cards Acceptance"
        assert result["objective"] == "Accept"
        assert result["metrics"] == {
            "lift": 0.0,
            "performance": 68.87,
            "performanceMeasure": "AUC",
        }

    @pytest.mark.asyncio
    async def test_get_models_internal(self, async_prediction, async_client):
        """_get_models is a plain async helper (not @api_method), called internally."""
        async_client.get.return_value = mock_prediction_describe
        result = await async_prediction._get_models()

        assert len(result) == 3
        # Default model
        assert result[0]["model_type"] == "Primary Model"
        assert result[0]["contextName"] == "NoContext"
        # Category model
        assert result[1]["model_type"] == "CategoryModel"
        # Supporting model
        assert result[2]["model_type"] == "Supporting model"

    @pytest.mark.asyncio
    async def test_get_notifications_returns_paginated_list(
        self,
        async_prediction,
        async_client,
    ):
        async_client.get.return_value = mock_response_notifications
        result = await async_prediction.get_notifications(return_df=False)
        assert isinstance(result, AsyncPaginatedList)

    @pytest.mark.asyncio
    async def test_get_notifications_as_df(
        self,
        async_prediction,
        async_client,
    ):
        async_client.get.return_value = mock_response_notifications
        async_client.request.return_value = mock_response_notifications
        result = await async_prediction.get_notifications(return_df=True)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 10)

    @pytest.mark.asyncio
    async def test_get_notifications_with_category(
        self,
        async_prediction,
        async_client,
    ):
        async_client.get.return_value = mock_response_notifications
        result = await async_prediction.get_notifications(category="Performance")
        assert isinstance(result, AsyncPaginatedList)

    @pytest.mark.asyncio
    async def test_get_champion_challengers_only_active(
        self,
        async_prediction,
        async_client,
    ):
        """All models in mock_prediction_describe are ACTIVE with no challenger percentages
        -> all fall into the `only_active` branch."""
        async_client.get.return_value = mock_prediction_describe
        result = await async_prediction.get_champion_challengers()

        # 3 models (1 default + 1 category + 1 supporting) -> 3 CCs
        assert len(result) == 3
        for cc in result:
            assert cc.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
            assert cc.challenger_model is None
            assert cc.active_model is not None

        # Category model entry carries its category; default/supporting do not
        categories = [cc.category for cc in result]
        assert "Retention" in categories
        assert categories.count(None) == 2

    @pytest.mark.asyncio
    async def test_get_champion_challengers_with_non_active(
        self,
        async_client,
    ):
        """Exercises the non_active branch (models with challengerPercentage)."""
        describe = {
            "context": [
                {
                    "contextName": "NoContext",
                    "defaultModels": [
                        {
                            "id": "ACTIVE_MODEL",
                            "label": "Active",
                            "role": "ACTIVE",
                            "type": "Adaptive model",
                            "componentName": "Active",
                            "modelingTechnique": "Adaptive model - Bayesian",
                        },
                        {
                            "id": "CHALLENGER_MODEL",
                            "label": "Challenger",
                            "role": "CHALLENGER",
                            "type": "Adaptive model",
                            "componentName": "Challenger",
                            "modelingTechnique": "Adaptive model - Bayesian",
                            "championPercentage": 70,
                            "challengerPercentage": 30,
                            "activeModel": "ACTIVE_MODEL",
                        },
                    ],
                    "categoryModels": [],
                },
            ],
            "supportingModels": [],
        }
        async_client.get.return_value = describe
        prediction = AsyncPrediction(
            client=async_client,
            predictionId="P1",
            label="l",
            objective="o",
            subject="s",
            status="Completed",
            lastUpdateTime="20240718T120557.925 GMT",
        )
        result = await prediction.get_champion_challengers()

        # Two CCs: one from non_active (the CHALLENGER) and one from only_active
        # (the ACTIVE model, which has no champion/challenger percentages).
        assert len(result) == 2

        # Find the challenger entry and the only-active entry
        non_active_cc = next(cc for cc in result if cc.challenger_model is not None)
        only_active_cc = next(cc for cc in result if cc.challenger_model is None)

        # champion_percentage = 100 - challengerPercentage(30) = 70
        assert non_active_cc.champion_percentage == 70
        assert non_active_cc.challenger_model.model_id == "CHALLENGER_MODEL"
        assert non_active_cc.active_model.model_id == "ACTIVE_MODEL"
        assert non_active_cc.category is None
        assert non_active_cc.context == "NoContext"

        assert only_active_cc.active_model.model_id == "ACTIVE_MODEL"

    @pytest.mark.asyncio
    async def test_add_conditional_model(self, async_prediction, async_client):
        """add_conditional_model posts, then resolves the new CC via get_champion_challengers."""
        async_client.post.return_value = {"referenceID": "M-6011"}
        async_client.get.return_value = mock_prediction_describe

        result = await async_prediction.add_conditional_model(
            new_model="@baseclass!testModel_falcons",
            category="Retention",
        )

        async_client.post.assert_awaited_once_with(
            "prweb/api/PredictionStudio/v4/predictions/"
            "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS/"
            "category/Retention/models/@baseclass!testModel_falcons",
            data={"contextName": "NoContext"},
        )
        assert result is not None
        assert result.active_model.model_id.lower() == "@baseclass!testmodel_falcons"
        assert result.category == "Retention"

    @pytest.mark.asyncio
    async def test_add_conditional_model_with_model_object(
        self,
        async_prediction,
        async_client,
    ):
        """When `new_model` is an AsyncModel, its model_id is extracted."""
        from pdstools.infinity.resources.prediction_studio.v24_2.model import AsyncModel

        async_client.post.return_value = {"referenceID": "M-6011"}
        async_client.get.return_value = mock_prediction_describe

        model_obj = AsyncModel(
            client=async_client,
            modelId="@baseclass!testModel_falcons",
            label="testModel_falcons",
            modelType="Adaptive model",
            status="Completed",
        )
        result = await async_prediction.add_conditional_model(
            new_model=model_obj,
            category="Retention",
            context="NoContext",
        )
        assert result is not None
        assert result.active_model.model_id.lower() == "@baseclass!testmodel_falcons"

    @pytest.mark.asyncio
    async def test_add_conditional_model_post_failure_raises(
        self,
        async_prediction,
        async_client,
    ):
        """When _a_post raises PegaException, wrap in PegaMLopsError."""
        from httpx import Response

        from pdstools.infinity.internal._exceptions import PegaException, PegaMLopsError

        async def _raise(*args, **kwargs):
            exc = PegaException.__new__(PegaException)
            Exception.__init__(exc, "boom")
            exc.override_message = "boom"
            raise exc

        async_client.post.side_effect = _raise
        del Response  # silence unused import; Response not actually needed

        with pytest.raises(PegaMLopsError, match="Error when adding Conditional model"):
            await async_prediction.add_conditional_model(
                new_model="@baseclass!testModel_falcons",
                category="Retention",
            )

    @pytest.mark.asyncio
    async def test_add_conditional_model_empty_response_raises(
        self,
        async_prediction,
        async_client,
    ):
        """Falsy response from _a_post triggers PegaMLopsError."""
        from pdstools.infinity.internal._exceptions import PegaMLopsError

        async_client.post.return_value = None

        with pytest.raises(PegaMLopsError, match="Add conditional model failed"):
            await async_prediction.add_conditional_model(
                new_model="@baseclass!testModel_falcons",
                category="Retention",
            )

    @pytest.mark.asyncio
    async def test_get_staged_changes(self, async_prediction, async_client):
        mock_response = {
            "listOfChanges": [
                {
                    "change": "Added conditional model for the category Retention",
                    "type": "Model",
                    "ruleName": "testModel_falcons",
                    "caseID": "M-2042",
                    "updateDateTime": "20240718T120558.474 GMT",
                },
            ],
        }
        async_client.get.return_value = mock_response
        result = await async_prediction.get_staged_changes()

        assert len(result) == 1
        assert result[0]["change"] == "Added conditional model for the category Retention"

    @pytest.mark.asyncio
    async def test_package_staged_changes(self, async_prediction, async_client):
        mock_response = {
            "message": "Changes are packaged...",
            "referenceID": "M-3001",
        }
        async_client.post.return_value = mock_response
        result = await async_prediction.package_staged_changes()
        assert result["referenceID"] == "M-3001"


# ---------------------------------------------------------------------------
# AsyncModel
# ---------------------------------------------------------------------------


class TestAsyncModel:
    @pytest.mark.asyncio
    async def test_init_attributes(self, async_model):
        assert async_model.model_id == "@BASECLASS!TESTMODEL_FALCONS"
        assert async_model.label == "testModel_falcons"
        assert async_model.model_type == "Adaptive model"
        assert async_model.modeling_technique == "Adaptive model - Bayesian"
        assert async_model.source == "Pega"
        assert async_model.status == "Completed"
        assert isinstance(async_model.last_update_time, datetime.datetime)
        assert async_model.updated_by == "Somnath Paul"

    @pytest.mark.asyncio
    async def test_describe(self, async_model, async_client):
        async_client.get.return_value = mock_model_describe
        result = await async_model.describe()

        async_client.get.assert_awaited_once_with(
            "/prweb/api/PredictionStudio/v1/models/@BASECLASS!TESTMODEL_FALCONS",
        )
        assert result["modelId"] == "@BASECLASS!TESTMODEL_FALCONS"
        assert result["status"] == "Completed"
        assert result["label"] == "testModel_falcons"
        assert result["type"] == "Adaptive Model"
        assert result["metrics"] == {
            "performance": 0.0,
            "performanceMeasure": "AUC",
        }

    @pytest.mark.asyncio
    async def test_get_notifications_returns_paginated_list(
        self,
        async_model,
        async_client,
    ):
        mock_notif = {
            "notifications": [
                {
                    "description": "Model performance is at its minimum of 50 AUC",
                    "modelType": "Adaptive Model",
                    "notificationType": "Performance",
                    "notificationID": "N-3",
                    "notificationMnemonic": "Model performance is at its minimum",
                    "context": "SomeContext",
                    "impact": "High",
                    "triggerTime": "20240720T050406.219 GMT",
                },
            ],
        }
        async_client.get.return_value = mock_notif
        result = await async_model.get_notifications(return_df=False)
        assert isinstance(result, AsyncPaginatedList)

    @pytest.mark.asyncio
    async def test_repr(self, async_model):
        r = repr(async_model)
        assert "AsyncModel" in r
        assert "@BASECLASS!TESTMODEL_FALCONS" in r


# ---------------------------------------------------------------------------
# Cross-cutting: async mixin methods work through AsyncAPIResource
# ---------------------------------------------------------------------------


class TestAsyncMixinIntegration:
    """Verify that @api_method methods on mixins are native coroutines
    when accessed through async concrete classes.
    """

    def test_get_model_categories_is_coroutinefunction(self):
        import inspect

        assert inspect.iscoroutinefunction(AsyncPredictionStudio.get_model_categories)

    def test_upload_model_is_coroutinefunction(self):
        import inspect

        assert inspect.iscoroutinefunction(AsyncPredictionStudio.upload_model)

    def test_model_describe_is_coroutinefunction(self):
        import inspect

        assert inspect.iscoroutinefunction(AsyncModel.describe)

    def test_prediction_describe_is_coroutinefunction(self):
        import inspect

        assert inspect.iscoroutinefunction(AsyncPrediction.describe)

    def test_prediction_get_staged_changes_is_coroutinefunction(self):
        import inspect

        assert inspect.iscoroutinefunction(AsyncPrediction.get_staged_changes)

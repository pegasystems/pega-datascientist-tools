from unittest.mock import AsyncMock, MagicMock

import pytest
from pdstools.infinity.resources.prediction_studio.v26_1.model_upload import UploadedModel as UploadedModelV26
from pdstools.infinity.resources.prediction_studio.v26_1.prediction_studio import (
    AsyncPredictionStudio as AsyncStudioV26,
)
from pdstools.infinity.resources.prediction_studio.v26_1.prediction_studio import PredictionStudio as StudioV26
from pdstools.infinity.resources.prediction_studio.v27_1.model_upload import UploadedModel as UploadedModelV27
from pdstools.infinity.resources.prediction_studio.v27_1.prediction_studio import (
    AsyncPredictionStudio as AsyncStudioV27,
)
from pdstools.infinity.resources.prediction_studio.v27_1.prediction_studio import PredictionStudio as StudioV27


@pytest.mark.parametrize(
    ("studio_class", "uploaded_class", "version"),
    [(StudioV26, UploadedModelV26, "v1"), (StudioV27, UploadedModelV27, "v5")],
)
def test_sync_upload_uses_versioned_endpoint_and_resource(studio_class, uploaded_class, version, tmp_path):
    model_file = tmp_path / "model.pmml"
    model_file.write_bytes(b"model")
    model = MagicMock()
    model.get_file_path.return_value = str(model_file)
    client = MagicMock()
    client.post.return_value = {"repositoryName": "Repo", "filePath": "model.pmml"}

    uploaded = studio_class(client=client).upload_model(model, "model.pmml")

    model.validate.assert_called_once_with()
    client.post.assert_called_once_with(
        f"/prweb/api/PredictionStudio/{version}/model",
        data={"fileSource": "bW9kZWw=\n", "fileName": "model.pmml"},
    )
    assert type(uploaded) is uploaded_class


@pytest.mark.parametrize(
    ("studio_class", "uploaded_class", "version"),
    [(AsyncStudioV26, UploadedModelV26, "v1"), (AsyncStudioV27, UploadedModelV27, "v5")],
)
async def test_async_upload_uses_versioned_endpoint_and_resource(studio_class, uploaded_class, version, tmp_path):
    model_file = tmp_path / "model.pmml"
    model_file.write_bytes(b"model")
    model = MagicMock()
    model.get_file_path.return_value = str(model_file)
    client = AsyncMock()
    client.post.return_value = {"repositoryName": "Repo", "filePath": "model.pmml"}

    uploaded = await studio_class(client=client).upload_model(model, "model.pmml")

    model.validate.assert_called_once_with()
    client.post.assert_awaited_once_with(
        f"/prweb/api/PredictionStudio/{version}/model",
        data={"fileSource": "bW9kZWw=\n", "fileName": "model.pmml"},
    )
    assert type(uploaded) is uploaded_class

"""Model routes shared by API versions with the same response shape."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pdstools.infinity.resources.prediction_studio.schemas import ModelData
from pdstools.infinity.resources.prediction_studio.v25_1.model import (
    AsyncModel as AsyncModelV25,
)
from pdstools.infinity.resources.prediction_studio.v25_1.model import Model as ModelV25
from pdstools.infinity.resources.prediction_studio.v26_1.model import (
    AsyncModel as AsyncModelV26,
)
from pdstools.infinity.resources.prediction_studio.v26_1.model import Model as ModelV26
from pdstools.infinity.resources.prediction_studio.v27_1.model import (
    AsyncModel as AsyncModelV27,
)
from pdstools.infinity.resources.prediction_studio.v27_1.model import Model as ModelV27

MODEL_VERSIONS = [
    pytest.param(ModelV25, AsyncModelV25, "v2", "v1", id="v25"),
    pytest.param(ModelV26, AsyncModelV26, "v2", "v1", id="v26"),
    pytest.param(ModelV27, AsyncModelV27, "v5", "v5", id="v27"),
]
MODEL_DATA = {
    "modelId": "model",
    "label": "Test model",
    "modelType": "Adaptive model",
    "modelingTechnique": "Adaptive model - Bayesian",
    "status": "Active",
}


@pytest.mark.parametrize("sync_type,async_type,model_api,instances_api", MODEL_VERSIONS)
def test_sync_model_routes(sync_type, async_type, model_api, instances_api):
    client = MagicMock()
    client.get.return_value = {"modelId": "model"}
    model = sync_type(client=client, **MODEL_DATA)
    assert model.describe() == {"modelId": "model"}
    client.get.assert_called_once_with(f"/prweb/api/PredictionStudio/{model_api}/models/model")

    assert model.get_notifications()._url == (
        f"/prweb/api/PredictionStudio/{model_api}/models/model/notifications?category=All"
    )
    assert model.get_notifications(category="Performance")._url == (
        f"/prweb/api/PredictionStudio/{model_api}/models/model/notifications?category=Performance"
    )
    assert model.list_instances()._url == (f"/prweb/api/PredictionStudio/{instances_api}/models/model/instances")
    if sync_type is ModelV25:
        assert model._data_cls is ModelData


@pytest.mark.parametrize("sync_type,async_type,model_api,instances_api", MODEL_VERSIONS)
async def test_async_model_routes(sync_type, async_type, model_api, instances_api):
    client = MagicMock()
    client.get = AsyncMock(return_value={"modelId": "model"})
    model = async_type(client=client, **MODEL_DATA)
    assert await model.describe() == {"modelId": "model"}
    client.get.assert_awaited_once_with(f"/prweb/api/PredictionStudio/{model_api}/models/model")

    assert (await model.get_notifications())._url == (
        f"/prweb/api/PredictionStudio/{model_api}/models/model/notifications?category=All"
    )
    assert (await model.get_notifications(category="Performance"))._url == (
        f"/prweb/api/PredictionStudio/{model_api}/models/model/notifications?category=Performance"
    )
    assert (await model.list_instances())._url == (
        f"/prweb/api/PredictionStudio/{instances_api}/models/model/instances"
    )
    if async_type is AsyncModelV25:
        assert model._data_cls is ModelData

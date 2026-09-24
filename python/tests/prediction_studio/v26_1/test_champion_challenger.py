"""Version-specific routing and upload behavior for the shared champion mixin."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pdstools.infinity.resources.prediction_studio.v25_1 import (
    AsyncChampionChallenger as AsyncChampionChallengerv25,
)
from pdstools.infinity.resources.prediction_studio.v25_1 import ChampionChallenger as ChampionChallengerv25
from pdstools.infinity.resources.prediction_studio.v26_1.champion_challenger import (
    AsyncChampionChallenger as AsyncChampionChallengerv26,
)
from pdstools.infinity.resources.prediction_studio.v26_1.champion_challenger import (
    ChampionChallenger as ChampionChallengerv26,
)
from pdstools.infinity.resources.prediction_studio.v26_1.model_upload import (
    UploadedModel as UploadedModelv26,
)
from pdstools.infinity.resources.prediction_studio.v27_1.champion_challenger import (
    AsyncChampionChallenger as AsyncChampionChallengerv27,
)
from pdstools.infinity.resources.prediction_studio.v27_1.champion_challenger import (
    ChampionChallenger as ChampionChallengerv27,
)
from pdstools.infinity.resources.prediction_studio.v27_1.model_upload import (
    UploadedModel as UploadedModelv27,
)

VERSIONS = [
    (ChampionChallengerv26, AsyncChampionChallengerv26, UploadedModelv26, UploadedModelv27, "v4", "v1"),
    (ChampionChallengerv27, AsyncChampionChallengerv27, UploadedModelv27, UploadedModelv26, "v5", "v5"),
]


def test_v25_reuses_v26_champion_challenger_resources():
    assert ChampionChallengerv25 is ChampionChallengerv26
    assert AsyncChampionChallengerv25 is AsyncChampionChallengerv26


def make_resource(resource_type, client=None):
    active_model = MagicMock()
    active_model.model_id = "active"
    active_model.component_name = "component"
    challenger_model = MagicMock()
    challenger_model.model_id = "challenger"
    return resource_type(
        client=client or MagicMock(),
        prediction_id="prediction",
        active_model=active_model,
        challenger_model=challenger_model,
        cc_id="operation",
    )


@pytest.mark.parametrize("sync_type,async_type,upload_type,other_upload,api,predictor_api", VERSIONS)
def test_endpoint_table(sync_type, async_type, upload_type, other_upload, api, predictor_api):
    endpoints = make_resource(sync_type)._endpoints
    base = "/prweb/api/PredictionStudio"
    assert endpoints.operations("operation") == f"{base}/{api}/predictions/operations/operation"
    assert endpoints.delete_challenger("prediction", "challenger") == (
        f"{base}/{api}/predictions/prediction/models/challenger/Remove"
    )
    assert endpoints.promote_challenger("prediction", "challenger") == (
        f"{base}/{api}/predictions/prediction/models/challenger/Promote"
    )
    assert endpoints.update_pattern("prediction", "challenger") == (
        f"{base}/{api}/predictions/prediction/models/challenger/updatePattern"
    )
    assert endpoints.distribution("prediction", "active") == (
        f"{base}/{api}/predictions/prediction/models/active/distribution"
    )
    assert endpoints.predictor_add("prediction", "active") == (
        f"{base}/{predictor_api}/predictions/prediction/models/active/predictor/add"
    )
    assert endpoints.predictor_remove("prediction", "active") == (
        f"{base}/{predictor_api}/predictions/prediction/models/active/predictor/remove"
    )
    assert endpoints.component("prediction", "component") == (
        f"{base}/{api}/predictions/prediction/component/component"
    )
    assert endpoints.component_clone("prediction", "component") == (
        f"{base}/{api}/predictions/prediction/component/component/clone"
    )


@pytest.mark.parametrize("sync_type,async_type,upload_type,other_upload,api,predictor_api", VERSIONS)
def test_sync_predictor_routes(sync_type, async_type, upload_type, other_upload, api, predictor_api):
    client = MagicMock()
    client.patch.return_value = {"message": "ok"}
    resource = make_resource(sync_type, client)
    resource.add_predictor("Age", "numeric", ".Age", "Double", is_active_model=True)
    assert client.patch.call_args.args[0] == (
        f"/prweb/api/PredictionStudio/{predictor_api}/predictions/prediction/models/active/predictor/add"
    )
    assert client.patch.call_args.kwargs["data"]["predictorName"] == "Age"
    resource.remove_predictor("Age", parameterized=True)
    assert client.patch.call_args.args[0] == (
        f"/prweb/api/PredictionStudio/{predictor_api}/predictions/prediction/models/active/predictor/remove"
    )


@pytest.mark.parametrize("sync_type,async_type,upload_type,other_upload,api,predictor_api", VERSIONS)
async def test_async_status_route(sync_type, async_type, upload_type, other_upload, api, predictor_api):
    client = MagicMock()
    client.get = AsyncMock(return_value={"ModelUpdateStatus": "Approved"})
    resource = make_resource(async_type, client)
    assert await resource._status() == {"ModelUpdateStatus": "Approved"}
    client.get.assert_awaited_once_with(
        f"/prweb/api/PredictionStudio/{api}/predictions/operations/operation",
    )


@pytest.mark.parametrize("sync_type,async_type,upload_type,other_upload,api,predictor_api", VERSIONS)
def test_uploaded_model_type_remains_version_specific(
    sync_type,
    async_type,
    upload_type,
    other_upload,
    api,
    predictor_api,
):
    resource = make_resource(sync_type)
    resource.cc_id = None
    resource._a_post = AsyncMock(return_value={"referenceID": "operation"})
    resource._check_then_update = AsyncMock(return_value={"message": "Approved"})
    resource._refresh_champion_challenger = AsyncMock()
    resource._sleep = AsyncMock()
    upload = upload_type("repository", "model.pmml")
    resource.add_model(upload, challenger_response_share=0.2)
    assert resource._a_post.call_args.args[0] == (
        f"/prweb/api/PredictionStudio/{api}/predictions/prediction/component/component"
    )
    assert resource._a_post.call_args.kwargs["data"]["sourceType"] == "Uploaded Model"

    other = other_upload("repository", "model.pmml")
    resource.add_model(other, challenger_response_share=0.2)
    assert resource._a_post.call_args.kwargs["data"]["sourceType"] == "Existing Model"

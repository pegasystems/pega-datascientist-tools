from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ....internal._resource import api_method
from ..base import AsyncPredictionStudioBase, PredictionStudioBase
from .prediction import AsyncPrediction, Prediction
from .repository import AsyncRepository, Repository


class _PredictionStudioV24_1Mixin:
    """v24.1 PredictionStudio business logic — shared parts."""

    version: str = "24.1"

    @api_method
    async def upload_model(self, model, file_name):
        raise NotImplementedError()


class PredictionStudio(_PredictionStudioV24_1Mixin, PredictionStudioBase):
    def list_predictions(self) -> PaginatedList[Prediction]:
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        return PaginatedList(
            Prediction, self._client, "get", endpoint, _root="predictions"
        )

    def repository(self) -> Repository:
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/repository"
        response = self._client.get(endpoint)
        return Repository(client=self._client, **response)


class AsyncPredictionStudio(_PredictionStudioV24_1Mixin, AsyncPredictionStudioBase):
    async def list_predictions(self) -> AsyncPaginatedList[AsyncPrediction]:
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        return AsyncPaginatedList(
            AsyncPrediction, self._client, "get", endpoint, _root="predictions"
        )

    async def repository(self) -> AsyncRepository:
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/repository"
        response = await self._a_get(endpoint)
        return AsyncRepository(client=self._client, **response)

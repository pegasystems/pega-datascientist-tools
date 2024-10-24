from ....internal._pagination import PaginatedList
from ..base import AsyncPredictionStudioBase, PredictionStudioBase
from .prediction import Prediction
from .repository import Repository


class PredictionStudio(PredictionStudioBase):
    version: str = "24.1"

    def list_predictions(self) -> PaginatedList[Prediction]:
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        return PaginatedList(
            Prediction, self._client, "get", endpoint, _root="predictions"
        )

    def repository(self) -> Repository:
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/repository"
        response = self._client.get(endpoint)
        return Repository(client=self._client, **response)

    def upload_model(self, model, file_name):
        raise NotImplementedError()


class AsyncPredictionStudio(AsyncPredictionStudioBase):
    version: str = "24.1"

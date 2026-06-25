from __future__ import annotations


from .....internal._exceptions import NoMonitoringExportError, PegaException
from .....internal._pagination import AsyncPaginatedList
from ...base import AsyncNotification
from ...v24_1 import AsyncPredictionStudio as AsyncPredictionStudioPrevious
from ..datamart_export import AsyncDatamartExport
from ..model import AsyncModel
from ..prediction import AsyncPrediction
from ..repository import AsyncRepository
from ._mixin import _PredictionStudiov26_1Mixin
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import polars as pl

    from ...types import NotificationCategory


class AsyncPredictionStudio(_PredictionStudiov26_1Mixin, AsyncPredictionStudioPrevious):
    version: str = "26.1"

    async def repository(self) -> AsyncRepository:
        """Gets information about the repository from Prediction Studio.

        Returns
        -------
        AsyncRepository
            A simple object with the repository's details.

        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/repository"
        response = await self._a_get(endpoint)
        return AsyncRepository(
            client=self._client,
            repository_name=response["repositoryName"],
            type=response["repositoryType"],
            bucket_name=response["bucketName"],
            root_path=response["rootPath"],
            datamart_export_location=response["datamartExportLocation"],
        )

    @property
    def models(self) -> AsyncPaginatedList[AsyncModel]:
        """All models, addressable by label or id.

        Returns
        -------
        AsyncPaginatedList[AsyncModel]
            A lazily-fetched, mapping-style collection. Supports
            ``await ps.models.get(label='My Model')`` and
            ``async for m in ps.models``.
        """
        endpoint = "/prweb/api/PredictionStudio/v2/models"
        return AsyncPaginatedList(
            AsyncModel,
            self._client,
            "get",
            endpoint,
            _root="models",
            pageSize=100,
        )

    @property
    def predictions(self) -> AsyncPaginatedList[AsyncPrediction]:  # type: ignore[override]  # AsyncPaginatedList is invariant; v26 returns a prediction subtype.
        """All predictions, addressable by label or id.

        Returns
        -------
        AsyncPaginatedList[AsyncPrediction]
            A lazily-fetched, mapping-style collection. Supports
            ``await ps.predictions.get(label='My Prediction')`` and
            ``async for p in ps.predictions``.
        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions"
        return AsyncPaginatedList(
            AsyncPrediction,
            self._client,
            "get",
            endpoint,
            _root="predictions",
            pageSize=100,
        )

    async def list_models(
        self,
        return_df: bool = False,
    ) -> AsyncPaginatedList[AsyncModel] | pl.DataFrame:
        """Fetches a list of all models from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncModel] or polars.DataFrame

        """
        pages = self.models
        if not return_df:
            return pages
        return await pages.as_df()

    async def list_predictions(  # type: ignore[override]  # intentionally widens parent signature with return_df
        self,
        return_df: bool = False,
    ) -> AsyncPaginatedList[AsyncPrediction] | pl.DataFrame:
        """Fetches a list of all predictions from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncPrediction] or polars.DataFrame

        """
        pages = self.predictions
        if not return_df:
            return pages
        return await pages.as_df()

    async def get_prediction(
        self,
        prediction_id: str | None = None,
        label: str | None = None,
        **kwargs,
    ) -> AsyncPrediction:
        """Finds and returns a specific prediction from Prediction Studio.

        Parameters
        ----------
        prediction_id : str, optional
            The unique ID of the prediction.
        label : str, optional
            The label of the prediction.

        Returns
        -------
        AsyncPrediction

        """
        uniques = kwargs if kwargs else {}
        if prediction_id:
            uniques["prediction_id"] = prediction_id
        if label:
            uniques["label"] = label
        pages = cast("AsyncPaginatedList[AsyncPrediction]", await self.list_predictions())
        prediction = await pages.get(**uniques)
        if prediction is None:
            raise KeyError(f"No prediction found for lookup {uniques!r}")
        return prediction

    async def get_model(
        self,
        model_id: str | None = None,
        label: str | None = None,
        **kwargs,
    ) -> AsyncModel:
        """Finds and returns a specific model from Prediction Studio.

        Parameters
        ----------
        model_id : str, optional
            The unique ID of the model.
        label : str, optional
            The label of the model.

        Returns
        -------
        AsyncModel

        """
        uniques = kwargs if kwargs else {}
        if model_id:
            uniques["model_id"] = model_id.upper()
        if label:
            uniques["label"] = label
        pages = cast("AsyncPaginatedList[AsyncModel]", await self.list_models())
        model = await pages.get(**uniques)
        if model is None:
            raise KeyError(f"No model found for lookup {uniques!r}")
        return model

    async def trigger_datamart_export(self) -> AsyncDatamartExport:
        """Initiates an export of model data to the Repository.

        Returns
        -------
        AsyncDatamartExport
            An object with information about the data export process.

        """
        endpoint = "/prweb/api/PredictionStudio/v1/datamart/export"
        try:
            response = await self._a_post(endpoint)
        except NoMonitoringExportError as e:
            raise e
        except PegaException as e:
            raise ValueError("Error while triggering data mart export" + str(e)) from e
        return AsyncDatamartExport(client=self._client, **response)

    async def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: bool = False,
    ) -> AsyncPaginatedList[AsyncNotification] | pl.DataFrame:
        """Fetches a list of notifications from Prediction Studio.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None, optional
            The category of notifications to retrieve.
        return_df : bool, default False
            If True, returns as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncNotification] or polars.DataFrame

        """
        endpoint = "/prweb/api/PredictionStudio/v2/notifications"
        if category is None:
            category = "All"

        endpoint = f"{endpoint}?category={category}"

        notifications: AsyncPaginatedList[AsyncNotification] = AsyncPaginatedList(
            AsyncNotification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
            pageSize=100,
        )
        if return_df:
            return await notifications.as_df()
        return notifications

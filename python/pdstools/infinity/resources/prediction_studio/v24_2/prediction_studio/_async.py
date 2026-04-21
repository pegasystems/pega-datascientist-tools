from __future__ import annotations

import polars as pl

from .....internal._exceptions import NoMonitoringExportError, PegaException
from .....internal._pagination import AsyncPaginatedList
from ...base import AsyncNotification
from ...types import NotificationCategory
from ...v24_1 import AsyncPredictionStudio as AsyncPredictionStudioPrevious
from ..datamart_export import AsyncDatamartExport
from ..model import AsyncModel
from ..prediction import AsyncPrediction
from ..repository import AsyncRepository
from ._mixin import _PredictionStudioV24_2Mixin


class AsyncPredictionStudio(_PredictionStudioV24_2Mixin, AsyncPredictionStudioPrevious):
    version: str = "24.2"

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
        endpoint = "/prweb/api/PredictionStudio/v1/models"
        pages: AsyncPaginatedList[AsyncModel] = AsyncPaginatedList(
            AsyncModel,
            self._client,
            "get",
            endpoint,
            _root="models",
        )
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
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        pages: AsyncPaginatedList[AsyncPrediction] = AsyncPaginatedList(
            AsyncPrediction,
            self._client,
            "get",
            endpoint,
            _root="predictions",
        )
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
        pages = await self.list_predictions()
        return await pages.get(**uniques)  # type: ignore[union-attr, return-value]  # return_df defaults False so pages is AsyncPaginatedList

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
        pages = await self.list_models()
        return await pages.get(**uniques)  # type: ignore[union-attr, return-value]  # return_df defaults False so pages is AsyncPaginatedList

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
        endpoint = "prweb/api/PredictionStudio/v1/notifications"
        if category is None:
            category = "All"

        endpoint = f"{endpoint}?category={category}"

        notifications: AsyncPaginatedList[AsyncNotification] = AsyncPaginatedList(
            AsyncNotification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
        )
        if return_df:
            return await notifications.as_df()
        return notifications

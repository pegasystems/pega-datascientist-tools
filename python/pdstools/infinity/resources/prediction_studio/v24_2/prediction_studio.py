import base64
from typing import Literal, Optional, Union, overload

import polars as pl

from ....internal._exceptions import NoMonitoringExportError, PegaException
from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ....internal._resource import api_method
from ..base import AsyncNotification, LocalModel, Notification
from ..local_model_utils import ONNXModel
from ..types import NotificationCategory
from ..v24_1 import AsyncPredictionStudio as AsyncPredictionStudioPrevious
from ..v24_1 import PredictionStudio as PredictionStudioPrevious
from .datamart_export import AsyncDatamartExport, DatamartExport
from .model import AsyncModel, Model
from .model_upload import UploadedModel
from .prediction import AsyncPrediction, Prediction
from .repository import AsyncRepository, Repository


class _PredictionStudioV24_2Mixin:
    """v24.2 PredictionStudio business logic — shared parts."""

    version: str = "24.2"

    @api_method
    async def upload_model(self, model: LocalModel, file_name: str) -> UploadedModel:
        """
        Uploads a model to the repository.

        This function handles the uploading of a model to the Repository,
        creating an UploadedModel object for further MLops processes.

        Parameters
        ----------
        model : str or ONNX model object
            The model to be uploaded. This can be specified either as a file
            path (str) to the model file or as an ONNX model object.
        file_name : str
            The name of the file (including extension) to be uploaded to
            the repository.

        Returns
        -------
        UploadedModel
            Details about the uploaded model, including API response data.

        Raises
        ------
        ValueError
            Raised if an attempt is made to upload an ONNX model without
            the required conversion library.
        ModelValidationError
            If the model validation fails.
        """
        endpoint = "/prweb/api/PredictionStudio/v1/model"
        model.validate()
        if isinstance(model, ONNXModel):
            file_encode = base64.encodebytes(model._model.SerializeToString()).decode(
                "ascii"
            )
        else:
            with open(model.get_file_path(), "rb") as file:
                file_encode = base64.encodebytes(file.read()).decode("ascii")

        data = {"fileSource": file_encode, "fileName": file_name}
        response = await self._a_post(endpoint, data=data)
        return UploadedModel(
            repository_name=response["repositoryName"], file_path=response["filePath"]
        )

    @api_method
    async def get_model_categories(self):
        """
        Gets a list of model categories from Prediction Studio.

        This function gives you back a list where each item tells you about
        one type of model.  Which can be useful for adding a conditional
        model to a prediction.

        Returns
        -------
        list of dict
            Each dictionary in the list has details about a model category,
            like its name and other useful information.
        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/modelCategories"
        response = await self._a_get(endpoint)
        return [item for item in response["categories"]]


# -- Concrete sync class -----------------------------------------------------


class PredictionStudio(_PredictionStudioV24_2Mixin, PredictionStudioPrevious):
    version: str = "24.2"

    def repository(self) -> Repository:
        """
        Gets information about the repository from Prediction Studio.

        Returns
        -------
        Repository
            A simple object with the repository's details, ready to use.
        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/repository"
        response = self._client.get(endpoint)
        return Repository(
            client=self._client,
            repository_name=response["repositoryName"],
            type=response["repositoryType"],
            bucket_name=response["bucketName"],
            root_path=response["rootPath"],
            datamart_export_location=response["datamartExportLocation"],
        )

    @overload
    def list_models(
        self, return_df: Literal[False] = False
    ) -> PaginatedList[Model]: ...

    @overload
    def list_models(self, return_df: Literal[True]) -> pl.DataFrame: ...

    def list_models(
        self, return_df: bool = False
    ) -> Union[PaginatedList[Model], pl.DataFrame]:
        """
        Fetches a list of all models from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default False.

        Returns
        -------
        PaginatedList[Model] or polars.DataFrame
            Returns a list of models or a DataFrame with model information.
        """
        endpoint = "/prweb/api/PredictionStudio/v1/models"
        pages = PaginatedList(Model, self._client, "get", endpoint, _root="models")
        if not return_df:
            return pages

        return pl.DataFrame([getattr(mod, "_public_dict") for mod in pages])

    @overload
    def list_predictions(
        self, return_df: Literal[False] = False
    ) -> PaginatedList[Prediction]: ...

    @overload
    def list_predictions(self, return_df: Literal[True]) -> pl.DataFrame: ...

    def list_predictions(
        self, return_df: bool = False
    ) -> Union[PaginatedList[Prediction], pl.DataFrame]:
        """
        Fetches a list of all predictions from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default False.

        Returns
        -------
        PaginatedList[Prediction] or polars.DataFrame
            Returns a list of predictions or a DataFrame.
        """
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        pages = PaginatedList(
            Prediction, self._client, "get", endpoint, _root="predictions"
        )

        if not return_df:
            return pages
        else:
            return pl.DataFrame([getattr(pred, "_public_dict") for pred in pages])

    def get_prediction(
        self,
        prediction_id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> Prediction:
        """
        Finds and returns a specific prediction from Prediction Studio.

        Parameters
        ----------
        prediction_id : str, optional
            The unique ID of the prediction.
        label : str, optional
            The label of the prediction.
        **kwargs : dict, optional
            Other details to narrow down the search.

        Returns
        -------
        Prediction
            The prediction that matches your search criteria.

        Raises
        ------
        ValueError
            If you don't provide an ID or label.
        """
        uniques = kwargs if kwargs else {}
        if prediction_id:
            uniques["prediction_id"] = prediction_id
        if label:
            uniques["label"] = label
        return self.list_predictions().get(**uniques)

    def get_model(
        self,
        model_id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> Model:
        """
        Finds and returns a specific model from Prediction Studio.

        Parameters
        ----------
        model_id : str, optional
            The unique ID of the model.
        label : str, optional
            The label of the model.
        **kwargs : dict, optional
            Other details to help find the model.

        Returns
        -------
        Model
            The model that matches your search.

        Raises
        ------
        ValueError
            If you don't provide an ID or label.
        """
        uniques = kwargs if kwargs else {}
        if model_id:
            uniques["model_id"] = model_id.upper()
        if label:
            uniques["label"] = label
        return self.list_models().get(**uniques)

    def trigger_datamart_export(self) -> DatamartExport:
        """
        Initiates an export of model data to the Repository.

        Returns
        -------
        DatamartExport
            An object with information about the data export process.
        """
        endpoint = "/prweb/api/PredictionStudio/v1/datamart/export"
        try:
            response = self._client.post(endpoint)
        except NoMonitoringExportError as e:
            raise e
        except PegaException as e:
            raise ValueError("Error while triggering data mart export" + str(e)) from e
        return DatamartExport(client=self._client, **response)

    @overload
    def get_notifications(
        self,
        category: Optional[NotificationCategory] = None,
        return_df: Literal[False] = False,
    ) -> PaginatedList[Notification]: ...

    @overload
    def get_notifications(
        self,
        category: Optional[NotificationCategory] = None,
        return_df: Literal[True] = True,
    ) -> pl.DataFrame: ...

    def get_notifications(
        self,
        category: Optional[NotificationCategory] = None,
        return_df: bool = False,
    ) -> Union[PaginatedList[Notification], pl.DataFrame]:
        """
        Fetches a list of notifications from Prediction Studio.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None, optional
            The category of notifications to retrieve.
        return_df : bool, default False
            If True, returns the notifications as a DataFrame.

        Returns
        -------
        PaginatedList[Notification] or polars.DataFrame
            A list of notifications or a DataFrame.
        """
        endpoint = "prweb/api/PredictionStudio/v1/notifications"
        if category is None:
            category = "All"

        endpoint = f"{endpoint}?category={category}"

        notifications = PaginatedList(
            Notification, self._client, "get", endpoint, _root="notifications"
        )
        if return_df:
            return pl.DataFrame(
                [
                    getattr(notification, "_public_dict")
                    for notification in notifications
                ]
            )
        else:
            return notifications


# -- Concrete async class ----------------------------------------------------


class AsyncPredictionStudio(_PredictionStudioV24_2Mixin, AsyncPredictionStudioPrevious):
    version: str = "24.2"

    async def repository(self) -> AsyncRepository:
        """
        Gets information about the repository from Prediction Studio.

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
        self, return_df: bool = False
    ) -> Union[AsyncPaginatedList[AsyncModel], pl.DataFrame]:
        """
        Fetches a list of all models from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncModel] or polars.DataFrame
        """
        endpoint = "/prweb/api/PredictionStudio/v1/models"
        pages = AsyncPaginatedList(
            AsyncModel, self._client, "get", endpoint, _root="models"
        )
        if not return_df:
            return pages
        return await pages.as_df()

    async def list_predictions(
        self, return_df: bool = False
    ) -> Union[AsyncPaginatedList[AsyncPrediction], pl.DataFrame]:
        """
        Fetches a list of all predictions from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncPrediction] or polars.DataFrame
        """
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        pages = AsyncPaginatedList(
            AsyncPrediction, self._client, "get", endpoint, _root="predictions"
        )
        if not return_df:
            return pages
        return await pages.as_df()

    async def get_prediction(
        self,
        prediction_id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> AsyncPrediction:
        """
        Finds and returns a specific prediction from Prediction Studio.

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
        return await pages.get(**uniques)

    async def get_model(
        self,
        model_id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> AsyncModel:
        """
        Finds and returns a specific model from Prediction Studio.

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
        return await pages.get(**uniques)

    async def trigger_datamart_export(self) -> AsyncDatamartExport:
        """
        Initiates an export of model data to the Repository.

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
        category: Optional[NotificationCategory] = None,
        return_df: bool = False,
    ) -> Union[AsyncPaginatedList[AsyncNotification], pl.DataFrame]:
        """
        Fetches a list of notifications from Prediction Studio.

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

        notifications = AsyncPaginatedList(
            AsyncNotification, self._client, "get", endpoint, _root="notifications"
        )
        if return_df:
            return await notifications.as_df()
        return notifications

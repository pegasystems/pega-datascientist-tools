import base64
from typing import Literal, Optional, Union, overload

import polars as pl

from ....internal._exceptions import PegaException, NoMonitoringExportError
from ....internal._pagination import PaginatedList
from ..base import Notification, LocalModel
from ..local_model_utils import ONNXModel
from ..v24_1 import AsyncPredictionStudio as AsyncPredictionStudioPrevious
from ..v24_1 import PredictionStudio as PredictionStudioPrevious
from .datamart_export import DatamartExport
from .model import Model
from .model_upload import UploadedModel
from .prediction import Prediction
from .repository import Repository
from ..types import NotificationCategory


class PredictionStudio(PredictionStudioPrevious):
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

        This function gets information about all the models stored in Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default, it's False, and you get a PaginatedList.

        Returns
        -------
        PaginatedList[Model] or polars.DataFrame
            Returns a list of models or a DataFrame with model information, based on the `return_df` parameter.
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
    ) -> Union[PaginatedList[Model], pl.DataFrame]:
        """
        Fetches a list of all predictions from Prediction Studio.

        This function gets information about all the predictions stored in Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default, it's False, and you get a list.

        Returns
        -------
        PaginatedList[Model] or polar.DataFrame
            Returns a list of models or a DataFrame with model information, based on the `as_df` parameter.
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
        self, prediction_id: Optional[str] = None, label: Optional[str] = None, **kwargs
    ) -> Prediction:
        """
        Finds and returns a specific prediction from Prediction Studio.

        This function looks for a prediction using its ID or label. You can also use other details to narrow down the search.
        It's useful when you need to get information about one particular prediction.

        Parameters
        ----------
        id : str, optional
            The unique ID of the prediction you're looking for. By default, it's None.
        label : str, optional
            The label of the prediction you're interested in. Also None by default.
        **kwargs : dict, optional
            Other details you can specify to help find the prediction.

        Returns
        -------
        Prediction
            The prediction that matches your search criteria.

        Raises
        ------
        ValueError
            If you don't provide an ID or label, it will tell you that you need at least one of them.
        """
        uniques = kwargs if kwargs else {}
        if prediction_id:
            uniques["prediction_id"] = prediction_id
        if label:
            uniques["label"] = label
        return self.list_predictions().get(**uniques)

    def get_model(
        self, model_id: Optional[str] = None, label: Optional[str] = None, **kwargs
    ) -> Model:
        """
        Finds and returns a specific model from Prediction Studio.

        This function searches for a model using its ID or label. You can also use other details to help find exactly what you're looking for.
        It's handy when you need details about a particular model.

        Parameters
        ----------
        id : str, optional
            The unique ID of the model you want to find.
        label : str, optional
            The label of the model you're interested in.
        **kwargs : dict, optional
            Other details you can specify to help find the model.

        Returns
        -------
        Model
            The model that matches your search.

        Raises
        ------
        ValueError
            If you don't provide an ID or label, it will tell you that you need at least one of them.
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

        This function begins the process of moving model data from Prediction Studio to the Repository.

        Returns
        -------
        DataMartExport
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

    def upload_model(self, model: LocalModel, file_name: str) -> UploadedModel:
        """
        Uploads a model to the repository.

        This function handles the uploading of a model to the Repository, creating an UploadedModel object for further MLops processes.

        Parameters
        ----------
        model : str or ONNX model object
            The model to be uploaded. This can be specified either as a file path (str) to the model file or as an ONNX model object.
        file_name : str
            The name of the file (including extension) to be uploaded to the repository.

        Returns
        -------
        UploadedModel
            Details about the uploaded model, including API response data.

        Raises
        ------
        ValueError
            Raised if an attempt is made to upload an ONNX model without the required conversion library.
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
        response = self._client.post(endpoint, data=data)
        return UploadedModel(
            repository_name=response["repositoryName"], file_path=response["filePath"]
        )

    def get_model_categories(self):
        """
        Gets a list of model categories from Prediction Studio.

        This function gives you back a list where each item tells you about one type of model.
        Which can be useful for adding a condional model to a prediction.

        Returns
        -------
        list of dict
            Each dictionary in the list has details about a model category, like its name and other useful information.
        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/modelCategories"
        response = self._client.get(endpoint)
        return [item for item in response["categories"]]

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

        This function retrieves notifications from Prediction Studio. You can filter these notifications by their category.
        Optionally, the notifications can be returned as a DataFrame for easier analysis and visualization.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None, optional
            The category of notifications to retrieve. If not specified, all notifications are fetched.
        return_df : bool, default False
            If True, returns the notifications as a DataFrame. Otherwise, returns a list.

        Returns
        -------
        PaginatedList[Notification] or polars.DataFrame
            A list of notifications or a DataFrame containing the notifications, depending on the value of `return_df`.
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


class AsyncPredictionStudio(AsyncPredictionStudioPrevious):
    version: str = "24.2"

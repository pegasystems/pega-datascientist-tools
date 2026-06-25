from __future__ import annotations

from typing import Literal, overload, TYPE_CHECKING


from .....internal._exceptions import NoMonitoringExportError, PegaException
from .....internal._pagination import PaginatedList
from ...base import Notification
from ...v24_1 import PredictionStudio as PredictionStudioPrevious
from ..datamart_export import DatamartExport
from ..model import Model
from ..prediction import Prediction
from ..repository import Repository
from ._mixin import _PredictionStudiov26_1Mixin

if TYPE_CHECKING:
    import polars as pl
    from ...types import NotificationCategory


class PredictionStudio(_PredictionStudiov26_1Mixin, PredictionStudioPrevious):
    version: str = "26.1"

    def repository(self) -> Repository:
        """Gets information about the repository from Prediction Studio.

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

    @property
    def models(self) -> PaginatedList[Model]:
        """All models, addressable by label or id.

        Returns
        -------
        PaginatedList[Model]
            A lazily-fetched, mapping-style collection. Supports
            ``ps.models['My Model']`` (by label or id),
            ``'My Model' in ps.models``, ``ps.models.keys()`` and iteration.
        """
        endpoint = "/prweb/api/PredictionStudio/v2/models"
        return PaginatedList(Model, self._client, "get", endpoint, _root="models", pageSize=100)

    @property
    def predictions(self) -> PaginatedList[Prediction]:
        """All predictions, addressable by label or id.

        Returns
        -------
        PaginatedList[Prediction]
            A lazily-fetched, mapping-style collection. Supports
            ``ps.predictions['My Prediction']`` (by label or id),
            ``'My Prediction' in ps.predictions``, ``ps.predictions.keys()``
            and iteration.
        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions"
        return PaginatedList(
            Prediction,
            self._client,
            "get",
            endpoint,
            _root="predictions",
            pageSize=100,
        )

    @overload
    def list_models(
        self,
        return_df: Literal[False] = False,
    ) -> PaginatedList[Model]: ...

    @overload
    def list_models(self, return_df: Literal[True]) -> pl.DataFrame: ...

    def list_models(
        self,
        return_df: bool = False,
    ) -> PaginatedList[Model] | pl.DataFrame:
        """Fetches a list of all models from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default False.

        Returns
        -------
        PaginatedList[Model] or polars.DataFrame
            Returns a list of models or a DataFrame with model information.

        """
        pages = self.models
        if not return_df:
            return pages

        return pages.as_df()

    @overload
    def list_predictions(
        self,
        return_df: Literal[False] = False,
    ) -> PaginatedList[Prediction]: ...

    @overload
    def list_predictions(self, return_df: Literal[True]) -> pl.DataFrame: ...

    def list_predictions(
        self,
        return_df: bool = False,
    ) -> PaginatedList[Prediction] | pl.DataFrame:
        """Fetches a list of all predictions from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default False.

        Returns
        -------
        PaginatedList[Prediction] or polars.DataFrame
            Returns a list of predictions or a DataFrame.

        """
        pages = self.predictions

        if not return_df:
            return pages
        return pages.as_df()

    def get_prediction(
        self,
        prediction_id: str | None = None,
        label: str | None = None,
        **kwargs,
    ) -> Prediction:
        """Finds and returns a specific prediction from Prediction Studio.

        Parameters
        ----------
        prediction_id : str, optional
            The unique ID of the prediction.
        label : str, optional
            The label of the prediction.

        Returns
        -------
        Prediction
            The prediction that matches your search criteria.

        Raises
        ------
        ValueError
            If you don't provide an ID or label, or if no prediction matches.

        """
        uniques = kwargs if kwargs else {}
        if prediction_id:
            uniques["prediction_id"] = prediction_id
        if label:
            uniques["label"] = label
        result = self.list_predictions().get(**uniques)
        if result is None:
            raise ValueError(f"No prediction matches the search criteria: {uniques}")
        return result

    def get_model(
        self,
        model_id: str | None = None,
        label: str | None = None,
        **kwargs,
    ) -> Model:
        """Finds and returns a specific model from Prediction Studio.

        Parameters
        ----------
        model_id : str, optional
            The unique ID of the model.
        label : str, optional
            The label of the model.

        Returns
        -------
        Model
            The model that matches your search.

        Raises
        ------
        ValueError
            If you don't provide an ID or label, or if no model matches.

        """
        uniques = kwargs if kwargs else {}
        if model_id:
            uniques["model_id"] = model_id.upper()
        if label:
            uniques["label"] = label
        result = self.list_models().get(**uniques)
        if result is None:
            raise ValueError(f"No model matches the search criteria: {uniques}")
        return result

    def trigger_datamart_export(self) -> DatamartExport:
        """Initiates an export of model data to the Repository.

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
        category: NotificationCategory | None = None,
        return_df: Literal[False] = False,
    ) -> PaginatedList[Notification]: ...

    @overload
    def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: Literal[True] = True,
    ) -> pl.DataFrame: ...

    def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: bool = False,
    ) -> PaginatedList[Notification] | pl.DataFrame:
        """Fetches a list of notifications from Prediction Studio.

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
        endpoint = "/prweb/api/PredictionStudio/v2/notifications"
        if category is None:
            category = "All"

        endpoint = f"{endpoint}?category={category}"

        notifications: PaginatedList[Notification] = PaginatedList(
            Notification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
            pageSize=100,
        )
        if return_df:
            return notifications.as_df()
        return notifications

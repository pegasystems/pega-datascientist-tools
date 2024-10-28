from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Literal, Optional, Union, overload

import polars as pl
from pydantic import validate_call

from .....utils import cdh_utils
from ....internal._constants import METRIC
from ....internal._exceptions import NoMonitoringInfo, PegaException, PegaMLopsError
from ....internal._pagination import PaginatedList
from ..base import Notification
from ..types import NotificationCategory
from ..v24_1.prediction import Prediction as PredictionPrevious
from .champion_challenger import ChampionChallenger
from .model import Model


class Prediction(PredictionPrevious):
    """
    The `Prediction` class provide functionality including retrieving notifications, models,
    adding conditional models, getting champion challengers, metrics, and plotting metrics.
    """

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
        Fetches a list of notifications for a specific prediction.

        This function retrieves notifications related to a prediction. You can filter these notifications by their category.
        Optionally, the notifications can be returned as a DataFrame for easier analysis and visualization.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None
            The category of notifications to retrieve. If not specified, all notifications are fetched.
        return_df : bool, default False
            If True, returns the notifications as a DataFrame. Otherwise, returns a list.

        Returns
        -------
        PaginatedList[Notification] or polars.DataFrame
            A list of notifications or a DataFrame containing the notifications, depending on the value of `return_df`.
        """

        endpoint = f"prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/notifications"
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

    def _get_models(self) -> List[Dict[str, str]]:
        """
        Internel function to fetch models linked to a specific prediction.

        This function gathers all models that are connected to a particular prediction.
        It organizes these models into three groups: default models, category models, and supporting models,
        each serving a unique role in making the prediction.


        Returns
        -------
        list of PredictionModel
            A collection of `PredictionModel` objects, each representing a model associated with the prediction.
        """
        models_list = []
        prediction_details = self.describe()
        # Extract default models from context and create PredictionModel instances
        for context in prediction_details.get("context", []):
            for model_detail in context.get("defaultModels", []):
                model = dict(**model_detail)
                model["model_type"] = "Primary Model"
                model["contextName"] = context.get("contextName")
                model["prediction_id"] = self.prediction_id
                models_list.append(model)
            for model_detail in context.get("categoryModels", []):
                model = dict(**model_detail)
                model["model_type"] = "CategoryModel"
                model["contextName"] = context.get("contextName")
                model["prediction_id"] = self.prediction_id
                models_list.append(model)
        for model_detail in prediction_details.get("supportingModels", []):
            model = dict(**model_detail)
            model["model_type"] = "Supporting model"
            model["prediction_id"] = self.prediction_id
            models_list.append(model)
        return models_list

    def get_champion_challengers(self) -> List[ChampionChallenger]:
        """
        Fetches list of ChampionChallenger objects linked to the prediction.

        This function fetches Champion-challenger pairs from a prediction.
        In cases where a challenger model is absent, it returns a ChampionChallenger object containing only the champion model.

        Returns
        -------
        list of ChampionChallenger
            A list of entries, each pairing a primary model with its challenger across various segments of the prediction.
        """
        ccs = []
        models = self._get_models()
        non_active = [
            mod for mod in models if mod["role"] not in {"ACTIVE", "CHAMPION"}
        ]

        only_active = [
            mod
            for mod in models
            if all(
                key not in mod for key in ["championPercentage", "challengerPercentage"]
            )
        ]

        for model in non_active:
            active_model_temp = {
                "modelId": model["id"],
                "label": model["label"],
                "modelType": model["type"],
                "status": model["role"],
                "componentName": model["componentName"],
                "modelingTechnique": model["modelingTechnique"],
            }
            ccs.append(
                ChampionChallenger(
                    client=self._client,
                    prediction_id=self.prediction_id,
                    champion_percentage=100 - model["challengerPercentage"],
                    challenger_model=Model(client=self._client, **active_model_temp),
                    context=model["contextName"],
                    category=model["categoryName"]
                    if model.get("categoryName") is not None
                    else None,
                    model_objective=model["model_type"],
                    active_model=[
                        Model(
                            client=self._client,
                            **{
                                "modelId": mod["id"],
                                "label": mod["label"],
                                "modelType": mod["type"],
                                "status": mod["role"],
                                "componentName": mod["componentName"],
                                "modelingTechnique": mod["modelingTechnique"]
                                if mod.get("modelingTechnique") is not None
                                else None,
                            },
                        )
                        for mod in models
                        if model["activeModel"] is not None
                        and mod["id"] == model["activeModel"]
                        and mod["contextName"] == model["contextName"]
                    ][0],
                )
            )

        for model in only_active:
            active_model_temp = {
                "modelId": model["id"],
                "label": model["label"],
                "modelType": model["type"],
                "status": model["role"],
                "componentName": model["componentName"],
                "modelingTechnique": model["modelingTechnique"]
                if model.get("modelingTechnique") is not None
                else None,
            }
            ccs.append(
                ChampionChallenger(
                    client=self._client,
                    prediction_id=self.prediction_id,
                    context=model["contextName"],
                    model_objective=model["model_type"],
                    category=model["categoryName"]
                    if model.get("categoryName") is not None
                    else None,
                    challenger_model=None,
                    active_model=Model(client=self._client, **active_model_temp),
                )
            )

        return ccs

    def add_conditional_model(
        self,
        new_model: Union[str, Model],
        category: str,
        context: Optional[str] = None,
    ) -> ChampionChallenger:
        """
        Incorporates a new model into a prediction for a specified category and context.

        This function allows for the addition of a new model to a prediction, tailored to a specific business category or use case. It enables the designation of a context for the model's application.

        Parameters
        ----------
        new_model : str
            Identifier of the model to be added.
        category : str
            The category under which the model will be classified.
        context : str, optional
            The specific context or scenario in which the model will be utilized.

        Returns
        -------
        ChampionChallenger
            An object detailing the updated configuration with the newly added model.
        """
        if isinstance(new_model, Model):
            new_model = new_model.model_id
        if context is None:
            context = "NoContext"
        endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/category/{category}/models/{new_model}"
        data = {}
        if context:
            data["contextName"] = context
        try:
            response = self._post(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError(
                "Error when adding Conditional model: " + str(e)
            ) from e
        if not response:
            raise PegaMLopsError("Add conditional model failed")
        champion_challengers = self.get_champion_challengers()
        for cc in champion_challengers:
            if cc.category is not None:
                if (
                    cc.active_model.model_id.lower() == new_model.lower()
                    and cc.context.lower() == context.lower()
                    and cc.category.lower() == category.lower()
                ):
                    return cc

    @validate_call
    def get_metric(
        self,
        *,
        metric: METRIC,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        frequency: Literal["Daily", "Weekly", "Monthly"] = "Daily",
    ) -> pl.DataFrame:
        """
        Fetches and returns metric data for a prediction within a specified date range, using datetime objects for dates.

        This method retrieves metric data, such as performance or usage statistics, for a given prediction. The data can be fetched for a specific period and at a defined frequency (daily, weekly, or monthly).

        Parameters
        ----------
        metric : METRIC
            The type of metric to retrieve.
        start_date : datetime
            The start date for the data retrieval.
        end_date : datetime, optional
            The end date for the data retrieval. If not provided, data is fetched until the current date.
        frequency : {"Daily", "Weekly", "Monthly"}, optional
            The frequency at which to retrieve the data. Defaults to "Daily".

        Returns
        -------
        pl.DataFrame
            A DataFrame containing the requested metric data, including values, snapshot times, and data usage.

        Raises
        ------
        NoMonitoringInfo
            If no monitoring data is available for the given parameters.
        """
        start_date_str = (
            start_date.strftime("%d/%m/%Y")
            if start_date
            else (date.today() - timedelta(days=7)).strftime("%d/%m/%Y")
        )
        end_date_str = end_date.strftime("%d/%m/%Y") if end_date else None

        endpoint = f"/prweb/api/PredictionStudio/v2/predictions/{self.prediction_id}/metric/{metric}"
        try:
            info = self._client.get(
                endpoint,
                startDate=start_date_str,
                endDate=end_date_str,
                frequency=frequency,
            )
            data = (
                pl.DataFrame(
                    info["monitoringData"],
                    schema={
                        "value": pl.Utf8,
                        "snapshotTime": pl.Utf8,
                        "dataUsage": pl.Utf8,
                    },
                )
                .with_columns(
                    snapshotTime=cdh_utils.parse_pega_date_time_formats(
                        "snapshotTime", "%Y-%m-%dT%H:%M:%S%.fZ"
                    ),
                    value=pl.col("value").replace("", None).cast(pl.Float64),
                    category=pl.col("dataUsage"),
                )
                .drop("dataUsage")
            )
            return data
        except NoMonitoringInfo:
            data = pl.DataFrame(
                schema={
                    "value": pl.Float64,
                    "snapshotTime": pl.Datetime("ns"),
                    "category": pl.Utf8,
                }
            )
            return data

    def package_staged_changes(self, message: Optional[str] = None):
        """
        Initiates the deployment of pending changes for a prediction model into the production setting.

        This function initiates a Change Request (CR) to either deploy pending changes directly to a Revision Manager,
        if available, or to create a branch with all pending changes in Prediction Studio.
        An optional message can be included to detail the nature of the changes being deployed.

        Parameters
        ----------
        message : str, optional
            Descriptive message about the changes being deployed. Defaults to "Approving the changes" if not specified.

        Returns
        -------
        Response
            Details the result of the deployment process.
        """
        endpoint = (
            f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/staged"
        )
        if message is None:
            message = "Approving the changes"
        data = {"reviewNote": message}
        try:
            response = self._client.post(endpoint, data=data)
        except PegaException as e:
            raise ValueError(
                "Error when Deploying Prediction changes: " + str(e)
            ) from e
        return response

    def get_staged_changes(self):
        """
        Retrieves a list of changes for a specific prediction.

        This method is used to fetch the list of changes that have been made to a prediction but not yet deployed to
        the production environment. The changes are staged and pending deployment.

        Returns:
            list: A list of changes staged for the prediction, detailing each modification pending deployment.
        """
        endpoint = (
            f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/staged"
        )
        responses = self._client.get(endpoint, data=None)
        return responses["listOfChanges"]

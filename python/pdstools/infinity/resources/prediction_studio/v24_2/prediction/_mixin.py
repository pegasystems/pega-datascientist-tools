from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Literal
from collections.abc import Callable

import polars as pl
from pydantic import validate_call

from ......utils import cdh_utils
from .....internal._constants import METRIC
from .....internal._exceptions import NoMonitoringInfo, PegaException
from .....internal._resource import api_method


class _PredictionV24_2Mixin:
    """v24.2 Prediction business logic — shared parts."""

    # Declared for mypy — provided by concrete base classes at runtime
    if TYPE_CHECKING:
        prediction_id: str
        _a_get: Callable[..., Any]
        _a_post: Callable[..., Any]

    async def _get_models(self) -> list[dict[str, str]]:
        """Internal function to fetch models linked to a specific prediction.

        This function gathers all models that are connected to a particular
        prediction.  It organizes these models into three groups: default
        models, category models, and supporting models, each serving a
        unique role in making the prediction.

        Returns
        -------
        list of dict
            A collection of model dicts associated with the prediction.

        """
        models_list = []
        endpoint = f"/prweb/api/PredictionStudio/v3/predictions/{self.prediction_id}"
        prediction_details = await self._a_get(endpoint)
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

    @api_method
    @validate_call
    async def get_metric(
        self,
        *,
        metric: METRIC,
        start_date: date | None = None,
        end_date: date | None = None,
        frequency: Literal["Daily", "Weekly", "Monthly"] = "Daily",
    ) -> pl.DataFrame:
        """Fetches and returns metric data for a prediction within a specified
        date range, using datetime objects for dates.

        This method retrieves metric data, such as performance or usage
        statistics, for a given prediction. The data can be fetched for a
        specific period and at a defined frequency (daily, weekly, or monthly).

        Parameters
        ----------
        metric : METRIC
            The type of metric to retrieve.
        start_date : datetime
            The start date for the data retrieval.
        end_date : datetime, optional
            The end date for the data retrieval. If not provided, data is
            fetched until the current date.
        frequency : {"Daily", "Weekly", "Monthly"}, optional
            The frequency at which to retrieve the data. Defaults to "Daily".

        Returns
        -------
        pl.DataFrame
            A DataFrame containing the requested metric data, including
            values, snapshot times, and data usage.

        Raises
        ------
        NoMonitoringInfo
            If no monitoring data is available for the given parameters.

        """
        start_date_str = (
            start_date.strftime("%d/%m/%Y") if start_date else (date.today() - timedelta(days=7)).strftime("%d/%m/%Y")
        )
        end_date_str = end_date.strftime("%d/%m/%Y") if end_date else None

        endpoint = f"/prweb/api/PredictionStudio/v2/predictions/{self.prediction_id}/metric/{metric}"
        try:
            info = await self._a_get(
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
                        "snapshotTime",
                        "%Y-%m-%dT%H:%M:%S%.fZ",
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
                },
            )
            return data

    @api_method
    async def package_staged_changes(self, message: str | None = None):
        """Initiates the deployment of pending changes for a prediction model
        into the production setting.

        This function initiates a Change Request (CR) to either deploy pending
        changes directly to a Revision Manager, if available, or to create a
        branch with all pending changes in Prediction Studio.  An optional
        message can be included to detail the nature of the changes being
        deployed.

        Parameters
        ----------
        message : str, optional
            Descriptive message about the changes being deployed. Defaults to
            "Approving the changes" if not specified.

        Returns
        -------
        dict
            Details the result of the deployment process.

        """
        endpoint = f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/staged"
        if message is None:
            message = "Approving the changes"
        data = {"reviewNote": message}
        try:
            response = await self._a_post(endpoint, data=data)
        except PegaException as e:
            raise ValueError(
                "Error when Deploying Prediction changes: " + str(e),
            ) from e
        return response

    @api_method
    async def get_staged_changes(self):
        """Retrieves a list of changes for a specific prediction.

        This method is used to fetch the list of changes that have been made
        to a prediction but not yet deployed to the production environment.
        The changes are staged and pending deployment.

        Returns
        -------
        list
            A list of changes staged for the prediction, detailing each
            modification pending deployment.

        """
        endpoint = f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/staged"
        responses = await self._a_get(endpoint, data=None)
        return responses["listOfChanges"]

from typing import Literal, overload

import polars as pl

from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ....internal._resource import api_method
from ..base import AsyncModel as AsyncPreviousModel
from ..base import AsyncNotification, ModelAttributes, Notification
from ..base import Model as PreviousModel
from ..types import NotificationCategory


class _ModelV24_2Mixin:
    """v24.2 Model business logic — defined once."""

    @api_method
    async def describe(self) -> ModelAttributes:
        """Fetches details about a model, including target labels, alternate labels, monitoring information, predictor information etc.

        Returns
        -------
        ModelAttributes
            An object containing information about the model.

        """
        endpoint = f"/prweb/api/PredictionStudio/v1/models/{self.model_id}"
        return await self._a_get(endpoint)


class Model(_ModelV24_2Mixin, PreviousModel):
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
        """Fetches a list of notifications for a specific model.

        This function retrieves notifications related to a model. You can filter these notifications by their category.
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
        endpoint = f"prweb/api/PredictionStudio/v1/models/{self.model_id}/notifications"
        if category is None:
            category = "All"

        endpoint = f"{endpoint}?category={category}"
        notifications = PaginatedList(
            Notification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
        )
        if return_df:
            return pl.DataFrame(
                [
                    getattr(notification, "_public_dict", {})
                    for notification in notifications
                ],
            )
        return notifications


class AsyncModel(_ModelV24_2Mixin, AsyncPreviousModel):
    async def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: bool = False,
    ) -> AsyncPaginatedList[AsyncNotification] | pl.DataFrame:
        """Fetches a list of notifications for a specific model.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None, optional
            The category of notifications to retrieve. If not specified, all notifications are fetched.
        return_df : bool, default False
            If True, returns the notifications as a DataFrame. Otherwise, returns a list.

        Returns
        -------
        AsyncPaginatedList[AsyncNotification] or polars.DataFrame
            A list of notifications or a DataFrame containing the notifications.

        """
        endpoint = f"prweb/api/PredictionStudio/v1/models/{self.model_id}/notifications"
        if category is None:
            category = "All"

        endpoint = f"{endpoint}?category={category}"
        notifications = AsyncPaginatedList(
            AsyncNotification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
        )
        if return_df:
            return await notifications.as_df()
        return notifications

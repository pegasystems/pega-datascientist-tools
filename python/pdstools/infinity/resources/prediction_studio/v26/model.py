from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ....internal._resource import api_method
from ..base import AsyncModel as AsyncPreviousModel
from ..base import AsyncNotification, ModelAttributes, Notification
from ..base import Model as PreviousModel

if TYPE_CHECKING:
    from ..types import NotificationCategory
    from collections.abc import Callable


class _Modelv26Mixin:
    """v26 Model business logic — defined once."""

    # Declared for mypy — provided by concrete base classes at runtime
    if TYPE_CHECKING:
        model_id: str
        _a_get: Callable[..., Any]

    def __init__(
        self,
        client,
        *,
        modelId: str,
        label: str,
        modelType: str,
        status: str,
        componentName: str | None = None,
        source: str | None = None,
        lastUpdateTime: str | None = None,
        modelingTechnique: str | None = None,
        updatedBy: str | None = None,
        performance: float | None = None,
        performanceMeasure: str | None = None,
        **kwargs,
    ):
        super().__init__(  # type: ignore[call-arg]
            client=client,
            modelId=modelId,
            label=label,
            modelType=modelType,
            status=status,
            componentName=componentName,
            source=source,
            lastUpdateTime=lastUpdateTime,
            modelingTechnique=modelingTechnique,
            updatedBy=updatedBy,
        )
        self.performance = performance
        self.performance_measure = performanceMeasure

    @api_method
    async def describe(self) -> ModelAttributes:
        """Fetches details about a model.

        Returns
        -------
        ModelAttributes
            An object containing information about the model.

        """
        endpoint = f"/prweb/api/PredictionStudio/v2/models/{self.model_id}"
        return await self._a_get(endpoint)


class Model(_Modelv26Mixin, PreviousModel):
    """v26 Model — inherits all v24.2 functionality."""

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
        endpoint = f"/prweb/api/PredictionStudio/v2/models/{self.model_id}/notifications"
        if category is None:
            category = "All"

        endpoint = f"{endpoint}?category={category}"
        notifications: PaginatedList[Notification] = PaginatedList(
            Notification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
        )
        if return_df:
            return pl.DataFrame(
                [getattr(notification, "_public_dict", {}) for notification in notifications],
            )
        return notifications


class AsyncModel(_Modelv26Mixin, AsyncPreviousModel):
    """v26 async Model — inherits all v24.2 functionality."""

    async def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: bool = False,
    ) -> AsyncPaginatedList[AsyncNotification] | pl.DataFrame:
        """Fetches a list of notifications for a specific model.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None, optional
            The category of notifications to retrieve.
        return_df : bool, default False
            If True, returns the notifications as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncNotification] or polars.DataFrame
            A list of notifications or a DataFrame.

        """
        endpoint = f"/prweb/api/PredictionStudio/v2/models/{self.model_id}/notifications"
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

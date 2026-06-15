from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ....internal._resource import api_method
from ..base import AsyncModel as AsyncPreviousModel
from ..base import (
    AsyncModelInstance,
    AsyncNotification,
    ModelAttributes,
    ModelInstance,
    Notification,
)
from ..base import Model as PreviousModel
from ..schemas import ModelDataV26_1

if TYPE_CHECKING:
    from ..types import NotificationCategory
    from collections.abc import Callable


class _Modelv26_1Mixin:
    """v26 Model business logic — defined once."""

    # Declared for mypy — provided by concrete base classes at runtime
    if TYPE_CHECKING:
        model_id: str
        _a_get: Callable[..., Any]

    # Construction is handled by base ``_ModelMixin`` (payload -> _data_cls).
    # v26.1 payloads additionally carry ``performance`` / ``performanceMeasure``,
    # captured by the version-specific ``ModelDataV26_1`` schema.
    _data_cls = ModelDataV26_1

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


class Model(_Modelv26_1Mixin, PreviousModel):
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

    @overload
    def list_instances(
        self,
        return_df: Literal[False] = False,
    ) -> PaginatedList[ModelInstance]: ...

    @overload
    def list_instances(
        self,
        return_df: Literal[True] = True,
    ) -> pl.DataFrame: ...

    def list_instances(
        self,
        return_df: bool = False,
    ) -> PaginatedList[ModelInstance] | pl.DataFrame:
        """Fetches a list of model instances for a specific model.

        Parameters
        ----------
        return_df : bool, default False
            If True, returns the model instances as a DataFrame.

        Returns
        -------
        PaginatedList[ModelInstance] or polars.DataFrame
            A list of model instances or a DataFrame.

        """
        endpoint = f"/prweb/api/PredictionStudio/v1/models/{self.model_id}/instances"
        instances: PaginatedList[ModelInstance] = PaginatedList(
            ModelInstance,
            self._client,
            "get",
            endpoint,
            _root="instances",
        )
        if return_df:
            return pl.DataFrame(
                [instance._public_dict for instance in instances],
            )
        return instances


class AsyncModel(_Modelv26_1Mixin, AsyncPreviousModel):
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

    @overload
    def list_instances(
        self,
        return_df: Literal[False] = False,
    ) -> AsyncPaginatedList[AsyncModelInstance]: ...

    @overload
    def list_instances(
        self,
        return_df: Literal[True] = True,
    ) -> pl.DataFrame: ...

    async def list_instances(
        self,
        return_df: bool = False,
    ) -> AsyncPaginatedList[AsyncModelInstance] | pl.DataFrame:
        """Fetches a list of model instances for a specific model.

        Parameters
        ----------
        return_df : bool, default False
            If True, returns the model instances as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncModelInstance] or polars.DataFrame
            A list of model instances or a DataFrame.

        """
        endpoint = f"/prweb/api/PredictionStudio/v1/models/{self.model_id}/instances"
        instances: AsyncPaginatedList[AsyncModelInstance] = AsyncPaginatedList(
            AsyncModelInstance,
            self._client,
            "get",
            endpoint,
            _root="instances",
        )
        if return_df:
            return await instances.as_df()
        return instances

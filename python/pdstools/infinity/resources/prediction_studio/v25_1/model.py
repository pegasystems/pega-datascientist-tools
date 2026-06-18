from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload


from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ....internal._resource import api_method
from ..base import AsyncModel as AsyncPreviousModel
from ..base import AsyncNotification, ModelAttributes, Notification
from ..base import Model as PreviousModel

if TYPE_CHECKING:
    import polars as pl
    from ..types import NotificationCategory
    from collections.abc import Callable


class _Modelv25_1Mixin:
    """v25 Model business logic — defined once.

    Identical to v26 except there is no ``__init__`` override, so the base
    ``_ModelMixin.__init__`` is used and ``performance`` /
    ``performance_measure`` attributes are never set (they are absent from
    the v25 list-models API response).
    """

    # Declared for mypy — provided by concrete base classes at runtime
    if TYPE_CHECKING:
        model_id: str
        _a_get: Callable[..., Any]

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


class Model(_Modelv25_1Mixin, PreviousModel):
    """v25 Model — no performance fields."""

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
            return notifications.as_df()
        return notifications


class AsyncModel(_Modelv25_1Mixin, AsyncPreviousModel):
    """v25 async Model — no performance fields."""

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

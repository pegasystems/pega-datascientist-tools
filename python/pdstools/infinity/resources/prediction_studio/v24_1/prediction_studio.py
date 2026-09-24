from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ....internal._resource import api_method
from ..base import AsyncPredictionStudioBase, PredictionStudioBase
from .prediction import AsyncPrediction, Prediction
from .repository import AsyncRepository, Repository

if TYPE_CHECKING:
    import polars as pl


class _PredictionStudioV24_1Mixin:
    """v24.1 PredictionStudio business logic — shared parts."""

    version: str = "24.1"

    @api_method
    async def upload_model(self, model, file_name):
        raise NotImplementedError


class PredictionStudio(_PredictionStudioV24_1Mixin, PredictionStudioBase):
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
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        return PaginatedList(
            Prediction,
            self._client,
            "get",
            endpoint,
            _root="predictions",
        )

    @overload
    def list_predictions(self, return_df: Literal[False] = False) -> PaginatedList[Prediction]: ...

    @overload
    def list_predictions(self, return_df: Literal[True]) -> pl.DataFrame: ...

    def list_predictions(self, return_df: bool = False) -> PaginatedList[Prediction] | pl.DataFrame:
        pages = self.predictions
        if not return_df:
            return pages
        return pages.as_df()

    def repository(self) -> Repository:
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/repository"
        response = self._client.get(endpoint)
        return Repository(client=self._client, **response)


class AsyncPredictionStudio(_PredictionStudioV24_1Mixin, AsyncPredictionStudioBase):
    @property
    def predictions(self) -> AsyncPaginatedList[AsyncPrediction]:
        """All predictions, addressable by label or id.

        Returns
        -------
        AsyncPaginatedList[AsyncPrediction]
            A lazily-fetched, mapping-style collection. Supports
            ``await ps.predictions.get(label='My Prediction')`` and
            ``async for p in ps.predictions``.
        """
        endpoint = "/prweb/api/PredictionStudio/V2/predictions"
        return AsyncPaginatedList(
            AsyncPrediction,
            self._client,
            "get",
            endpoint,
            _root="predictions",
        )

    @overload
    async def list_predictions(self, return_df: Literal[False] = False) -> AsyncPaginatedList[AsyncPrediction]: ...

    @overload
    async def list_predictions(self, return_df: Literal[True]) -> pl.DataFrame: ...

    async def list_predictions(self, return_df: bool = False) -> AsyncPaginatedList[AsyncPrediction] | pl.DataFrame:
        pages = self.predictions
        if not return_df:
            return pages
        return await pages.as_df()

    async def repository(self) -> AsyncRepository:
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/repository"
        response = await self._a_get(endpoint)
        return AsyncRepository(client=self._client, **response)

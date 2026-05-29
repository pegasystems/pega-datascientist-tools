from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, overload

import polars as pl

from ....internal._pagination import AsyncPaginatedList, PaginatedList
from ..v26_1.prediction_studio._async import AsyncPredictionStudio as _AsyncPredictionStudiov26_1
from ..v26_1.prediction_studio._sync import PredictionStudio as _PredictionStudiov26_1
from .model import AsyncModel, Model
from .prediction import AsyncPrediction, Prediction

if TYPE_CHECKING:
    pass


class PredictionStudio(_PredictionStudiov26_1):
    """v25 PredictionStudio — overrides list_models and list_predictions only.

    Same endpoints as v26; the v25 API responses omit ``performance`` /
    ``performanceMeasure`` so v25-specific Model and Prediction classes are
    used to keep those attributes out of ``_public_dict`` / DataFrames.
    """

    version: str = "25.1"

    @overload
    def list_models(self, return_df: Literal[False] = False) -> PaginatedList[Model]: ...

    @overload
    def list_models(self, return_df: Literal[True]) -> pl.DataFrame: ...

    def list_models(self, return_df: bool = False) -> PaginatedList[Model] | pl.DataFrame:
        """Fetches a list of all models from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default False.

        Returns
        -------
        PaginatedList[Model] or polars.DataFrame

        """
        endpoint = "/prweb/api/PredictionStudio/v2/models"
        pages: PaginatedList[Model] = PaginatedList(
            Model, self._client, "get", endpoint, _root="models", pageSize=100
        )
        if not return_df:
            return pages
        return pl.DataFrame([mod._public_dict for mod in pages])

    @overload  # type: ignore[override]
    def list_predictions(self, return_df: Literal[False] = False) -> PaginatedList[Prediction]: ...

    @overload
    def list_predictions(self, return_df: Literal[True]) -> pl.DataFrame: ...

    def list_predictions(self, return_df: bool = False) -> PaginatedList[Prediction] | pl.DataFrame:
        """Fetches a list of all predictions from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame. By default False.

        Returns
        -------
        PaginatedList[Prediction] or polars.DataFrame

        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions"
        pages: PaginatedList[Prediction] = PaginatedList(
            Prediction, self._client, "get", endpoint, _root="predictions", pageSize=100
        )
        if not return_df:
            return pages
        return pl.DataFrame([pred._public_dict for pred in pages])


class AsyncPredictionStudio(_AsyncPredictionStudiov26_1):
    """v25 async PredictionStudio — overrides list_models and list_predictions only."""

    version: str = "25.1"

    async def list_models(self, return_df: bool = False) -> AsyncPaginatedList[AsyncModel] | pl.DataFrame:
        """Fetches a list of all models from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncModel] or polars.DataFrame

        """
        endpoint = "/prweb/api/PredictionStudio/v2/models"
        pages: AsyncPaginatedList[AsyncModel] = AsyncPaginatedList(
            AsyncModel, self._client, "get", endpoint, _root="models", pageSize=100
        )
        if not return_df:
            return pages
        return await pages.as_df()

    async def list_predictions(  # type: ignore[override]
        self, return_df: bool = False
    ) -> AsyncPaginatedList[AsyncPrediction] | pl.DataFrame:
        """Fetches a list of all predictions from Prediction Studio.

        Parameters
        ----------
        return_df : bool, optional
            Set to True to get the results as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncPrediction] or polars.DataFrame

        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions"
        pages: AsyncPaginatedList[AsyncPrediction] = AsyncPaginatedList(
            AsyncPrediction, self._client, "get", endpoint, _root="predictions", pageSize=100
        )
        if not return_df:
            return pages
        return await pages.as_df()

    async def get_prediction(
        self, prediction_id: str | None = None, label: str | None = None, **kwargs
    ) -> AsyncPrediction:
        """Finds and returns a specific prediction from Prediction Studio."""
        uniques = {**kwargs}
        if prediction_id:
            uniques["prediction_id"] = prediction_id
        if label:
            uniques["label"] = label
        pages = cast("AsyncPaginatedList[AsyncPrediction]", await self.list_predictions())
        return await pages.get(**uniques)

    async def get_model(
        self, model_id: str | None = None, label: str | None = None, **kwargs
    ) -> AsyncModel:
        """Finds and returns a specific model from Prediction Studio."""
        uniques = {**kwargs}
        if model_id:
            uniques["model_id"] = model_id.upper()
        if label:
            uniques["label"] = label
        pages = cast("AsyncPaginatedList[AsyncModel]", await self.list_models())
        return await pages.get(**uniques)

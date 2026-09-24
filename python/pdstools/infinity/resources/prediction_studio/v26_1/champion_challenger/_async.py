from __future__ import annotations

from typing import TYPE_CHECKING

from .....internal._pagination import AsyncPaginatedList
from ...base import AsyncChampionChallenger as AsyncChampionChallengerBase
from ..model import AsyncModel
from ._mixin import _ChampionChallengerv26_1Mixin

if TYPE_CHECKING:
    import polars as pl


class AsyncChampionChallenger(
    _ChampionChallengerv26_1Mixin,
    AsyncChampionChallengerBase,
):
    """v26 async ChampionChallenger — inherits all v24.2 functionality."""

    _replacement_options_version = "v1"
    _model_cls = AsyncModel

    async def list_available_models_to_add(
        self,
        return_df: bool = False,
    ) -> AsyncPaginatedList | pl.DataFrame:
        """Fetches a list of models eligible to be challengers.

        Parameters
        ----------
        return_df : bool, optional
            Determines the format of the returned data: a DataFrame if True,
            otherwise an async list of model instances. Defaults to False.

        Returns
        -------
        AsyncPaginatedList[AsyncModel] or pl.DataFrame
            An async list of model instances or a DataFrame of models.

        """
        endpoint = f"/prweb/api/PredictionStudio/{self._replacement_options_version}/predictions/{self.prediction_id}/component/{self.active_model.component_name}/replacement-options"
        pages: AsyncPaginatedList[AsyncModel] = AsyncPaginatedList(
            self._model_cls,
            self._client,
            "get",
            endpoint,
            _root="models",
        )
        if not return_df:
            return pages
        return await pages.as_df()

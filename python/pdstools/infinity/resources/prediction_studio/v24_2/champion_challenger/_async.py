from __future__ import annotations

import polars as pl

from .....internal._pagination import AsyncPaginatedList
from ...base import AsyncChampionChallenger as AsyncChampionChallengerBase
from ._mixin import _ChampionChallengerV24_2Mixin


class AsyncChampionChallenger(
    _ChampionChallengerV24_2Mixin,
    AsyncChampionChallengerBase,
):
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
        from ..model import AsyncModel

        endpoint = f"prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/component/{self.active_model.component_name}/replacement-options"
        pages: AsyncPaginatedList[AsyncModel] = AsyncPaginatedList(
            AsyncModel,
            self._client,
            "get",
            endpoint,
            _root="models",
        )
        if not return_df:
            return pages
        return await pages.as_df()

from __future__ import annotations

import polars as pl

from .....internal._pagination import PaginatedList
from ...base import ChampionChallenger as ChampionChallengerBase
from ._mixin import _ChampionChallengerV24_2Mixin


class ChampionChallenger(_ChampionChallengerV24_2Mixin, ChampionChallengerBase):
    def list_available_models_to_add(
        self,
        return_df: bool = False,
    ) -> PaginatedList | pl.DataFrame:
        """Fetches a list of models eligible to be challengers.

        Queries for models that can be added as challengers to the current
        prediction for the current active model. Offers the option to return
        the results in a DataFrame format for easier data handling.

        Parameters
        ----------
        return_df : bool, optional
            Determines the format of the returned data: a DataFrame if True,
            otherwise a list of model instances. Defaults to False.

        Returns
        -------
        PaginatedList[Model] or pl.DataFrame
            A list of model instances or a DataFrame of models, based on the
            ``return_df`` parameter choice.

        """
        from ..model import Model

        endpoint = f"prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/component/{self.active_model.component_name}/replacement-options"
        pages: PaginatedList[Model] = PaginatedList(Model, self._client, "get", endpoint, _root="models")
        if not return_df:
            return pages
        return pl.DataFrame([mod._public_dict for mod in pages])

from __future__ import annotations

from ...v26_1.champion_challenger import AsyncChampionChallenger as AsyncChampionChallengerV26_1
from ..model import AsyncModel
from ._mixin import _ChampionChallengerv27_1Mixin


class AsyncChampionChallenger(_ChampionChallengerv27_1Mixin, AsyncChampionChallengerV26_1):
    """v27 async ChampionChallenger with v5 replacement options."""

    _replacement_options_version = "v5"
    _model_cls = AsyncModel

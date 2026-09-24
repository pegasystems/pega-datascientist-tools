from __future__ import annotations

from ...v26_1.champion_challenger import ChampionChallenger as ChampionChallengerV26_1
from ..model import Model
from ._mixin import _ChampionChallengerv27_1Mixin


class ChampionChallenger(_ChampionChallengerv27_1Mixin, ChampionChallengerV26_1):
    """v27 ChampionChallenger with v5 replacement options."""

    _replacement_options_version = "v5"
    _model_cls = Model

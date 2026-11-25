from __future__ import annotations

from ...v26_1.champion_challenger._mixin import _ChampionChallengerv26_1Mixin, build_champion_challenger_endpoints
from ..model_upload import UploadedModel


class _ChampionChallengerv27_1Mixin(_ChampionChallengerv26_1Mixin):
    """Use v26 champion/challenger methods with v5 routing and uploads."""

    _endpoints = build_champion_challenger_endpoints("v5")
    _uploaded_model_type = UploadedModel

from __future__ import annotations

from ..._shared_champion_challenger import (
    _SharedChampionChallengerMixin,
    build_champion_challenger_endpoints,
)

# v26 kept the predictor add/remove endpoints on v1 while every other
# endpoint moved to v4 — the one observed case of a version not moving all
# of its endpoints in lockstep.
_ENDPOINTS = build_champion_challenger_endpoints("v4", predictor_api_version="v1")


class _ChampionChallengerv26_1Mixin(_SharedChampionChallengerMixin):
    """v26 ChampionChallenger — endpoints only; business logic is shared with v27_1.

    See ``_shared_champion_challenger.py`` for the full implementation and the
    rationale for why v24_2 is not part of this shared module.
    """

    _endpoints = _ENDPOINTS

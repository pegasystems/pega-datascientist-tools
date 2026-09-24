from __future__ import annotations

from ..._shared_champion_challenger import (
    _SharedChampionChallengerMixin,
    build_champion_challenger_endpoints,
)
from ..model_upload import UploadedModel

_ENDPOINTS = build_champion_challenger_endpoints("v5")


class _ChampionChallengerv27_1Mixin(_SharedChampionChallengerMixin):
    """v27 ChampionChallenger — endpoints only; business logic is shared with v26_1.

    See ``_shared_champion_challenger.py`` for the full implementation and the
    rationale for why v24_2 is not part of this shared module.
    """

    _endpoints = _ENDPOINTS
    _uploaded_model_type = UploadedModel

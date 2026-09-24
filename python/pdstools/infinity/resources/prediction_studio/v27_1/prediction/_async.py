from __future__ import annotations

from ...v26_1.prediction import AsyncPrediction as AsyncPredictionV26_1
from ..champion_challenger import AsyncChampionChallenger
from ..model import AsyncModel
from ._mixin import _Predictionv27_1Mixin


class AsyncPrediction(_Predictionv27_1Mixin, AsyncPredictionV26_1):
    """v27 async Prediction with v5 routing and v27 resource types."""

    _model_cls = AsyncModel
    _champion_challenger_cls = AsyncChampionChallenger

from __future__ import annotations

from ...v26_1.prediction import Prediction as PredictionV26_1
from ..champion_challenger import ChampionChallenger
from ..model import Model
from ._mixin import _Predictionv27_1Mixin


class Prediction(_Predictionv27_1Mixin, PredictionV26_1):
    """v27 Prediction with v5 routing and v27 resource types."""

    _model_cls = Model
    _champion_challenger_cls = ChampionChallenger

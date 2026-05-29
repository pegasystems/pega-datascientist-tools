from __future__ import annotations

# v25 has the same endpoints as v26 but list-models and list-predictions
# responses omit performance / performanceMeasure fields.  Override those
# two resources while re-exporting everything else unchanged from v26.
from ..v26_1 import (  # noqa: F401
    AsyncChampionChallenger,
    AsyncDatamartExport,
    AsyncRepository,
    ChampionChallenger,
    DatamartExport,
    Repository,
)
from .model import AsyncModel, Model
from .prediction import AsyncPrediction, Prediction
from .prediction_studio import AsyncPredictionStudio, PredictionStudio

__all__ = [
    "AsyncChampionChallenger",
    "AsyncDatamartExport",
    "AsyncModel",
    "AsyncPrediction",
    "AsyncPredictionStudio",
    "AsyncRepository",
    "ChampionChallenger",
    "DatamartExport",
    "Model",
    "Prediction",
    "PredictionStudio",
    "Repository",
]

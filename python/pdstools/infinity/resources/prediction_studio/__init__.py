from typing import Optional, Type

from . import v24_1, v24_2
from .base import AsyncPredictionStudioBase, PredictionStudioBase


def get(version: str) -> Optional[Type[PredictionStudioBase]]:
    if not version:
        return None
    if version == "24.1":
        return v24_1.PredictionStudio
    elif version == "24.2":
        return v24_2.PredictionStudio
    raise ValueError(version)


def get_async(version: str) -> Optional[Type[AsyncPredictionStudioBase]]:
    if not version:
        return None
    if version == "24.1":
        return v24_1.AsyncPredictionStudio
    elif version == "24.2":
        return v24_2.AsyncPredictionStudio
    raise ValueError(version)

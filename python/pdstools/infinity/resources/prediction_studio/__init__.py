import logging

from . import v24_1, v24_2
from .base import AsyncPredictionStudioBase, PredictionStudioBase

logger = logging.getLogger(__name__)


def get(version: str) -> type[PredictionStudioBase] | None:
    if not version:
        return None
    if version == "24.1":
        return v24_1.PredictionStudio
    if version == "24.2":
        return v24_2.PredictionStudio
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to the latest known API (24.2).",
        version,
    )
    return v24_2.PredictionStudio


def get_async(version: str) -> type[AsyncPredictionStudioBase] | None:
    if not version:
        return None
    if version == "24.1":
        return v24_1.AsyncPredictionStudio
    if version == "24.2":
        return v24_2.AsyncPredictionStudio
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to the latest known API (24.2).",
        version,
    )
    return v24_2.AsyncPredictionStudio

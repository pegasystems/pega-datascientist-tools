import logging

from . import v24_1, v24_2
from .base import AsyncPredictionStudioBase, PredictionStudioBase

logger = logging.getLogger(__name__)

_LATEST = v24_2

# Map Pega version strings to the Prediction Studio module that implements
# their API surface. Newer Pega versions that have not yet diverged from the
# 24.2 API are mapped to the latest known module so users don't have to pass
# a fake version string. Add a new entry (and a new vNN_M module) when the API
# surface changes.
_SYNC_VERSION_MAP: dict[str, type[PredictionStudioBase]] = {
    "24.1": v24_1.PredictionStudio,
    "24.2": v24_2.PredictionStudio,
    "25.1": _LATEST.PredictionStudio,
}

_ASYNC_VERSION_MAP: dict[str, type[AsyncPredictionStudioBase]] = {
    "24.1": v24_1.AsyncPredictionStudio,
    "24.2": v24_2.AsyncPredictionStudio,
    "25.1": _LATEST.AsyncPredictionStudio,
}


def get(version: str) -> type[PredictionStudioBase] | None:
    if not version:
        return None
    if version in _SYNC_VERSION_MAP:
        return _SYNC_VERSION_MAP[version]
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to the latest known API (24.2).",
        version,
    )
    return _LATEST.PredictionStudio


def get_async(version: str) -> type[AsyncPredictionStudioBase] | None:
    if not version:
        return None
    if version in _ASYNC_VERSION_MAP:
        return _ASYNC_VERSION_MAP[version]
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to the latest known API (24.2).",
        version,
    )
    return _LATEST.AsyncPredictionStudio

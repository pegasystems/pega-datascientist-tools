from __future__ import annotations

import logging

from . import v24_1, v24_2, v25, v26
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AsyncPredictionStudioBase, PredictionStudioBase

logger = logging.getLogger(__name__)

_LATEST = v26

_SYNC_VERSION_MAP: dict[str, type[PredictionStudioBase]] = {
    "24.1": v24_1.PredictionStudio,
    "24.2": v24_2.PredictionStudio,
    "25": v25.PredictionStudio,
    "25.1": v25.PredictionStudio,
    "26": v26.PredictionStudio,
}

_ASYNC_VERSION_MAP: dict[str, type[AsyncPredictionStudioBase]] = {
    "24.1": v24_1.AsyncPredictionStudio,
    "24.2": v24_2.AsyncPredictionStudio,
    "25": v25.AsyncPredictionStudio,
    "25.1": v25.AsyncPredictionStudio,
    "26": v26.AsyncPredictionStudio,
}


def get(version: str) -> type[PredictionStudioBase] | None:
    if not version:
        return None
    if version in _SYNC_VERSION_MAP:
        return _SYNC_VERSION_MAP[version]
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to the latest known API (26).",
        version,
    )
    return _LATEST.PredictionStudio


def get_async(version: str) -> type[AsyncPredictionStudioBase] | None:
    if not version:
        return None
    if version in _ASYNC_VERSION_MAP:
        return _ASYNC_VERSION_MAP[version]
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to the latest known API (26).",
        version,
    )
    return _LATEST.AsyncPredictionStudio

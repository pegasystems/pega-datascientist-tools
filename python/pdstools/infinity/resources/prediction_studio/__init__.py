from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import v24_1, v24_2, v25_1, v26_1, v27_1

if TYPE_CHECKING:
    from .base import AsyncPredictionStudioBase, PredictionStudioBase

logger = logging.getLogger(__name__)

_SYNC_VERSION_MAP: dict[str, type[PredictionStudioBase]] = {
    "24.1": v24_1.PredictionStudio,
    "24.2": v24_2.PredictionStudio,
    "25.1": v25_1.PredictionStudio,
    "26.1": v26_1.PredictionStudio,
    "27.1": v27_1.PredictionStudio,
}

_ASYNC_VERSION_MAP: dict[str, type[AsyncPredictionStudioBase]] = {
    "24.1": v24_1.AsyncPredictionStudio,
    "24.2": v24_2.AsyncPredictionStudio,
    "25.1": v25_1.AsyncPredictionStudio,
    "26.1": v26_1.AsyncPredictionStudio,
    "27.1": v27_1.AsyncPredictionStudio,
}


def _normalize_version(version: str) -> str:
    """Normalise a bare major version to its canonical ``<major>.<minor>`` form.

    Users sometimes pass ``"25"`` or ``"26"`` as a convenience shorthand.
    The canonical keys in the version map are ``"25.1"`` / ``"26.1"``, so we
    expand any dotless string by appending ``.1`` before lookup.
    """
    return version if "." in version else f"{version}.1"


def _fallback_version(version: str) -> str:
    """Keep the pre-v27 fallback for unrecognized older version strings."""
    parts = version.split(".")
    if len(parts) >= 2:
        try:
            major, minor = int(parts[0]), int(parts[1])
        except ValueError:
            return "26.1"
        if (major, minor) >= (27, 1):
            return "27.1"
    return "26.1"


def get(version: str) -> type[PredictionStudioBase]:
    version = _normalize_version(version)
    if version in _SYNC_VERSION_MAP:
        return _SYNC_VERSION_MAP[version]
    fallback = _fallback_version(version)
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to API %s.",
        version,
        fallback,
    )
    return _SYNC_VERSION_MAP[fallback]


def get_async(version: str) -> type[AsyncPredictionStudioBase]:
    version = _normalize_version(version)
    if version in _ASYNC_VERSION_MAP:
        return _ASYNC_VERSION_MAP[version]
    fallback = _fallback_version(version)
    logger.info(
        "Pega version '%s' is not explicitly supported; falling back to API %s.",
        version,
        fallback,
    )
    return _ASYNC_VERSION_MAP[fallback]

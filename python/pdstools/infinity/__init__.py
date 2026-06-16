"""Infinity API client for Pega Decision Management."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from ..utils.namespaces import MissingDependenciesException

if TYPE_CHECKING:
    from .client import AsyncInfinity, Infinity
    from .resources.prediction_studio.schemas import (
        ModelData,
        ModelInstanceData,
        NotificationData,
        PredictionData,
    )


class DependencyNotFound:
    def __init__(self, dependencies: list[str]):
        self.dependencies = dependencies
        self.namespace = "the DX API Client"
        self.deps_group = "api"

    def __repr__(self):
        return f"While importing, one or more dependencies were not found: {self.dependencies}"

    def __call__(self):
        raise MissingDependenciesException(
            self.dependencies,
            namespace=self.namespace,
            deps_group=self.deps_group,
        )


def _check_dependencies() -> list[str]:
    """Return list of missing dependencies for the Infinity client."""
    missing: list[str] = []
    if not find_spec("pydantic"):
        missing.append("pydantic")
    if not find_spec("httpx"):
        missing.append("httpx")
    return missing


_DATA_MODELS = (
    "ModelData",
    "ModelInstanceData",
    "NotificationData",
    "PredictionData",
)


def __getattr__(name: str):
    """Lazy import to avoid loading httpx / pydantic until needed."""
    if name in ("Infinity", "AsyncInfinity"):
        missing_dependencies = _check_dependencies()

        if missing_dependencies:
            return DependencyNotFound(missing_dependencies)

        from .client import AsyncInfinity, Infinity

        if name == "Infinity":
            return Infinity
        return AsyncInfinity

    if name in _DATA_MODELS:
        if not find_spec("pydantic"):
            return DependencyNotFound(["pydantic"])

        from .resources.prediction_studio.schemas import (
            ModelData,
            ModelInstanceData,
            NotificationData,
            PredictionData,
        )

        return {
            "ModelData": ModelData,
            "ModelInstanceData": ModelInstanceData,
            "NotificationData": NotificationData,
            "PredictionData": PredictionData,
        }[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AsyncInfinity",
    "Infinity",
    "ModelData",
    "ModelInstanceData",
    "NotificationData",
    "PredictionData",
]

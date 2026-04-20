"""Infinity API client for Pega Decision Management."""

from importlib.util import find_spec
from typing import TYPE_CHECKING

from ..utils.namespaces import MissingDependenciesException

if TYPE_CHECKING:
    from .client import AsyncInfinity, Infinity


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


def __getattr__(name: str):
    """Lazy import to avoid loading httpx until needed."""
    if name in ("Infinity", "AsyncInfinity"):
        missing_dependencies = _check_dependencies()

        if missing_dependencies:
            return DependencyNotFound(missing_dependencies)

        from .client import AsyncInfinity, Infinity

        if name == "Infinity":
            return Infinity
        return AsyncInfinity

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["AsyncInfinity", "Infinity"]

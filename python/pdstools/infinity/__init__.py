"""
Infinity API client for Pega Decision Management.
"""

from importlib.util import find_spec
from typing import TYPE_CHECKING, List

from ..utils.namespaces import MissingDependenciesException

if TYPE_CHECKING:
    from .client import Infinity


class DependencyNotFound:
    def __init__(self, dependencies: List[str]):
        self.dependencies = dependencies
        self.namespace = "the DX API Client"
        self.deps_group = "api"

    def __repr__(self):
        return f"While importing, one or more dependencies were not found: {self.dependencies}"

    def __call__(self):
        raise MissingDependenciesException(
            self.dependencies, namespace=self.namespace, deps_group=self.deps_group
        )


def __getattr__(name: str):
    """Lazy import to avoid loading httpx until needed."""
    if name == "Infinity":
        missing_dependencies: List[str] = []
        if not find_spec("pydantic"):
            missing_dependencies.append("pydantic")
        if not find_spec("httpx"):
            missing_dependencies.append("httpx")

        if missing_dependencies:
            return DependencyNotFound(missing_dependencies)

        from .client import Infinity

        return Infinity

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["Infinity"]
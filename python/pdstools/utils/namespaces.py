import importlib
import logging
import sys
import types
from functools import wraps
from typing import List, Optional

logger = logging.getLogger(__name__)


def require_dependencies(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.check_dependencies()
        return func(self, *args, **kwargs)

    return wrapper


class LazyNamespaceMeta(type):
    def __new__(cls, name, bases, dct):
        for attr_name, attr_value in dct.items():
            if isinstance(attr_value, types.FunctionType) and attr_name not in [
                "__init__",
                "check_dependencies",
                "_check_dependencies",
            ]:
                dct[attr_name] = require_dependencies(attr_value)
        return super().__new__(cls, name, bases, dct)


class LazyNamespace(metaclass=LazyNamespaceMeta):
    dependencies: Optional[List[str]]
    dependency_group: Optional[str]

    def __init__(self):
        self._dependencies_checked = False

    def check_dependencies(self):
        if not self._dependencies_checked:
            self._check_dependencies()
            self._dependencies_checked = True

    def _check_dependencies(self):
        not_installed = []
        if not hasattr(self, "dependencies") or not self.dependencies:
            return []
        for package in self.dependencies:
            if package in sys.modules:
                logger.debug(f"{package} is already imported.")
            elif importlib.util.find_spec(package) is not None:  # pragma: no cover
                logger.debug(f"{package} is installed, but not imported.")
            else:
                logger.debug(f"{package} is NOT installed.")
                not_installed.append(package)
        if not_installed:
            raise MissingDependenciesException(
                not_installed,
                self.__class__.__name__,
                self.dependency_group if hasattr(self, "dependency_group") else None,
            )
        return not_installed


class MissingDependenciesException(Exception):
    def __init__(
        self,
        deps: List[str],
        namespace: Optional[str] = None,
        deps_group: Optional[str] = None,
    ):
        if namespace:
            message = f"To use {namespace}, you are missing{' an' if len(deps)==1 else ''} optional {'dependency:' if len(deps)==1 else 'dependencies:'} {', '.join(deps)}."
        else:
            message = f"Missing dependencies: {deps}."

        message += f"\nPlease install {'it'if len(deps)==1 else 'them'} using your favorite package manager (e.g. uv pip install {' '.join(deps)})"

        if deps_group:
            message += f",\n\nNOTE: these dependencies are also shipped with the optional dependency group '{deps_group}', \nso you can install them automatically with pdstools as well (e.g. uv pip install pdstools['{deps_group}'])."
        else:
            message += "."

        self.message = message

        super().__init__(self.message)

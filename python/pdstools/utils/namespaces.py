import importlib
import logging
import sys
import types
from functools import wraps

logger = logging.getLogger(__name__)

# Mapping from import name (as used in ``LazyNamespace.dependencies`` or passed
# to :class:`MissingDependenciesException`) to the name used when installing
# the package via pip/uv.  Only needed where the two differ.
_IMPORT_TO_INSTALL_NAME: dict[str, str] = {
    "yaml": "pyyaml",
    "polars_hash": "polars-hash",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
}


def _to_install_name(dep: str) -> str:
    """Translate an import name to its pip/uv install name when they differ."""
    return _IMPORT_TO_INSTALL_NAME.get(dep, dep)


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
    dependencies: list[str] | None
    dependency_group: str | None

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
        deps: list[str],
        namespace: str | None = None,
        deps_group: str | None = None,
    ):
        install_names = [_to_install_name(d) for d in deps]

        if namespace:
            message = (
                f"To use {namespace}, you are missing"
                f"{' an' if len(deps) == 1 else ''} optional "
                f"{'dependency:' if len(deps) == 1 else 'dependencies:'} "
                f"{', '.join(deps)}."
            )
        else:
            message = f"Missing dependencies: {deps}."

        message += (
            f"\nPlease install {'it' if len(deps) == 1 else 'them'} using your "
            f"favorite package manager. Package name"
            f"{'' if len(deps) == 1 else 's'}: {' '.join(install_names)}"
        )

        if deps_group:
            message += (
                f".\n\nNOTE: these dependencies are also shipped with the optional "
                f"pdstools extra '{deps_group}', \nso you can install them "
                f"automatically by adding the 'pdstools[{deps_group}]' extra "
                f"with your package manager."
            )
        else:
            message += "."

        self.message = message
        self.deps = deps
        self.namespace = namespace
        self.deps_group = deps_group

        super().__init__(self.message)

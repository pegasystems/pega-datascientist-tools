from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from functools import wraps
from typing import ClassVar

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
    """Decorator that triggers dependency checking before invoking ``func``."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.check_dependencies()
        return func(self, *args, **kwargs)

    return wrapper


class LazyNamespaceMeta(type):
    """Lazy namespace meta."""

    def __new__(cls, name, bases, dct):
        """New."""
        for attr_name, attr_value in dct.items():
            if isinstance(attr_value, types.FunctionType) and attr_name not in [
                "__init__",
                "check_dependencies",
                "_check_dependencies",
                "_missing_dependencies",
                "__getattr__",
                "__repr__",
            ]:
                dct[attr_name] = require_dependencies(attr_value)
        return super().__new__(cls, name, bases, dct)


class LazyNamespace(metaclass=LazyNamespaceMeta):
    """Lazy namespace.

    Subclasses declare ``dependencies`` (a list of import names) and
    optionally ``dependency_group`` (the matching ``pdstools[<extra>]``
    extras group). Method calls trigger a lazy dependency check that
    raises :class:`MissingDependenciesException` with a friendly install
    hint when a required package is missing.

    Attribute access on missing methods (e.g. ``ns.some_method`` without
    calling) also triggers the dependency check, so introspection
    (``hasattr``, ``dir``) surfaces missing-dep errors uniformly instead
    of returning a confusing ``AttributeError``.
    """

    dependencies: ClassVar[list[str] | None] = None
    dependency_group: ClassVar[str | None] = None

    def __init__(self):
        self._dependencies_checked = False

    def check_dependencies(self):
        """Check dependencies."""
        if not self._dependencies_checked:
            self._check_dependencies()
            self._dependencies_checked = True

    def _missing_dependencies(self) -> list[str]:
        """Return missing dependency names without raising."""
        not_installed: list[str] = []
        if not getattr(self, "dependencies", None):
            return not_installed
        for package in self.dependencies or []:
            if package in sys.modules:
                continue
            if importlib.util.find_spec(package) is not None:  # pragma: no cover
                continue
            not_installed.append(package)
        return not_installed

    def _check_dependencies(self):
        not_installed = self._missing_dependencies()
        if not_installed:
            raise MissingDependenciesException(
                not_installed,
                self.__class__.__name__,
                self.dependency_group if hasattr(self, "dependency_group") else None,
            )
        return not_installed

    def __getattr__(self, name: str):
        # Only triggered when normal attribute lookup fails. Run the
        # dependency check first so ``hasattr(ns, "method_name")`` and
        # similar introspection produce the friendly missing-dep error
        # instead of a bare AttributeError that obscures the real cause.
        # Skip dunder/private to avoid interfering with object protocol
        # and to keep ``__init__``-time access to ``_dependencies_checked``
        # safe from infinite recursion. Use a re-entry guard so that
        # the dep-check itself (which reads ``dependency_group`` via
        # ``hasattr``/``getattr``) can't recurse into us.
        if name.startswith("_") or name in {
            "dependencies",
            "dependency_group",
            "check_dependencies",
        }:
            raise AttributeError(name)
        if object.__getattribute__(self, "__dict__").get("_in_lazyns_check"):
            raise AttributeError(name)
        try:
            object.__setattr__(self, "_in_lazyns_check", True)
            self.check_dependencies()
        finally:
            object.__setattr__(self, "_in_lazyns_check", False)
        raise AttributeError(name)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        try:
            missing = self._missing_dependencies()
        except Exception:  # pragma: no cover - defensive
            return f"<{cls_name}>"
        if missing:
            extras = f", install pdstools[{self.dependency_group}]" if getattr(self, "dependency_group", None) else ""
            return f"<{cls_name}: unavailable, missing {', '.join(missing)}{extras}>"
        return f"<{cls_name}>"


class MissingDependenciesException(Exception):
    """Missing dependencies exception."""

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

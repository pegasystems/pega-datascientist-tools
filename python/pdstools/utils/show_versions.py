from __future__ import annotations

import importlib
import re
import sys
from typing import Dict, Literal, Set, Union, overload

from .. import __version__

package_name = "pdstools"


@overload
def show_versions(print_output: Literal[True] = True) -> None: ...


@overload
def show_versions(print_output: Literal[False] = False) -> str: ...


def show_versions(print_output: bool = True) -> Union[None, str]:
    """
    Print or return version of pdstools and dependencies.

    Parameters
    ----------
    print_output : bool, optional
        If True, print the version information to stdout.
        If False, return the version information as a string.
        Default is True.

    Returns
    -------
    Optional[str]
        Version information as a string if print_output is False, else None.

    Examples
    --------
    >>> from pdstools import show_versions
    >>> show_versions()
    --- Version info ---
    pdstools: 4.0.0-alpha
    Platform: macOS-14.7-arm64-arm-64bit
    Python: 3.12.4 (main, Jun  6 2024, 18:26:44) [Clang 15.0.0 (clang-1500.3.9.4)]

    --- Dependencies ---
    typing_extensions: 4.12.2
    polars>=1.9: 1.9.0

    --- Dependency group: adm ---
    plotly>=5.5.0: 5.24.1

    --- Dependency group: api ---
    pydantic: 2.9.2
    httpx: 0.27.2
    """

    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    info = []
    info.append("--- Version info ---")
    info.append(f"{package_name}: {__version__}")
    info.append(f"Platform: {platform.platform()}")
    info.append(f"Python: {sys.version}")

    deps = grouped_dependencies()

    info.append("\n--- Dependencies ---")

    required = deps.pop("required")
    for d in required:
        info.append(f"{d}: {_get_dependency_version(d)}")

    for group, dependencies in deps.items():
        info.append(f"\n--- Dependency group: {group} ---")

        for d in dependencies:
            info.append(f"{d}: {_get_dependency_version(d)}")

    version_info = "\n".join(info)
    if print_output:
        print(version_info)
        return None
    else:
        return version_info


def expand_nested_deps(extras: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    def expand_dep(dep: str, processed: Set[str]) -> Set[str]:
        if not dep.startswith(f"{package_name}["):
            return {dep}

        nested_extras = dep.split("[")[1].split("]")[0].split(",")
        result = set()

        for nested_extra in nested_extras:
            if nested_extra in processed:
                continue  # pragma: no cover

            processed.add(nested_extra)
            if nested_extra in extras:
                result.update(
                    set().union(
                        *(expand_dep(d, processed.copy()) for d in extras[nested_extra])
                    )
                )

        return result if result else {dep}

    expanded = {}
    for extra, deps in extras.items():
        expanded[extra] = set().union(*(expand_dep(dep, set()) for dep in deps))

    return expanded


def grouped_dependencies() -> Dict[str, Set[str]]:
    extras: Dict[str, Set[str]] = {"required": set()}
    requires = importlib.metadata.distribution(package_name).requires
    if not requires:  # pragma: no cover
        return {}
    for dependency in requires:
        split = dependency.split("; extra == ")

        if len(split) == 1:
            extras["required"].add(split[0])
        else:
            package, extra = split
            extra = extra.strip('"')
            if extra not in extras:
                extras[extra] = set()
            extras[extra].add(package)

    return expand_nested_deps(extras)


def _get_dependency_version(dep_name: str) -> str:
    dep_name = re.sub("[<>=]", "|||", dep_name).split("|||")[0]
    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    else:
        # importlib.metadata was introduced in Python 3.8
        module_version = importlib.metadata.version(dep_name)

    return module_version

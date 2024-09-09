from __future__ import annotations

import importlib
import re
import sys
from typing import Dict, Optional, Set

from .. import __version__

package_name = "pdstools"


def show_versions() -> None:
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
    ---Version info---
    pdstools: 3.1.0
    Platform: macOS-12.6.4-x86_64-i386-64bit
    Python: 3.11.0 (v3.11.0:deaf509e8f, Oct 24 2022, 14:43:23) [Clang 13.0.0 (clang-1300.0.29.30)]

    ---Dependencies---
    plotly: 5.13.1
    requests: 2.28.1
    pydot: 1.4.2
    polars: 0.17.0
    pyarrow: 11.0.0.dev52
    tqdm: 4.64.1
    pyyaml: <not installed>
    aioboto3: 11.0.1

    ---Streamlit app dependencies---
    streamlit: 1.20.0
    quarto: 0.1.0
    papermill: 2.4.0
    itables: 1.5.1
    pandas: 1.5.3
    jinja2: 3.1.2
    xlsxwriter: 3.0
    """

    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    info = []
    info.append("--- Version info ---")
    info.append(f"pdstools: {__version__}")
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
    elif sys.version_info >= (3, 8):
        # importlib.metadata was introduced in Python 3.8
        module_version = importlib.metadata.version(dep_name)
    else:  # pragma: no cover
        module_version = "<version not detected>"

    return module_version

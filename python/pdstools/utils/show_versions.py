from __future__ import annotations

import importlib
import importlib.metadata
import sys
from typing import Optional

from .. import __version__

import importlib.metadata


def show_versions(print_output: bool = True) -> Optional[str]:
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
    info.append("---Version info---")
    info.append(f"pdstools: {__version__}")
    info.append(f"Platform: {platform.platform()}")
    info.append(f"Python: {sys.version}")

    info.append("\n---Dependencies---")
    deps = _get_dependency_info()
    for name, v in deps.items():
        info.append(f"{name}: {v}")

    info.append("\n---Streamlit app dependencies---")
    deps = _get_opt_dependency_info()
    for name, v in deps.items():
        info.append(f"{name}: {v}")
    version_info = "\n".join(info)
    if print_output:
        print(version_info)
        return None
    else:
        return version_info


def _get_dependency_info() -> dict[str, str]:
    # see the list of dependencies in pyproject.toml
    dependencies = [
        "plotly",
        "requests",
        "pydot",
        "polars",
        "pyarrow",
        "tqdm",
        "pyyaml",
        "aioboto3",
    ]
    return {name: _get_dependency_version(name) for name in dependencies}


def _get_opt_dependency_info() -> dict[str, str]:
    # see the list of dependencies in pyproject.toml
    opt_deps = [
        "streamlit",
        "quarto",
        "papermill",
        "itables",
        "pandas",
        "jinja2",
        "xlsxwriter",
    ]
    return {name: _get_dependency_version(name) for name in opt_deps}


def _get_dependency_version(dep_name: str) -> str:
    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    elif sys.version_info >= (3, 8):
        # importlib.metadata was introduced in Python 3.8
        module_version = importlib.metadata.version(dep_name)
    else:
        module_version = "<version not detected>"

    return module_version

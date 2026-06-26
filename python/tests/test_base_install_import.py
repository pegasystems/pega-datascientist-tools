"""Smoke tests for base ``import pdstools`` with optional deps absent."""

from __future__ import annotations

import subprocess
import sys

import pytest


OPTIONAL_IMPORTS = [
    "plotly",
    "pydot",
    "duckdb",
    "streamlit",
    "great_tables",
    "pydantic",
    "httpx",
    "aioboto3",
]


def _run_import_with_blocked_imports(blocked_imports: list[str]) -> subprocess.CompletedProcess[str]:
    blocked = ",".join(f'"{name}"' for name in blocked_imports)
    code = f"""
import builtins

orig_import = builtins.__import__
blocked = {{{blocked}}}

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked or any(name.startswith(pkg + ".") for pkg in blocked):
        raise ModuleNotFoundError(f"No module named '{{name.split('.')[0]}}'")
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
try:
    import pdstools  # noqa: F401
finally:
    builtins.__import__ = orig_import
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("missing_dep", OPTIONAL_IMPORTS)
def test_import_pdstools_with_each_optional_dependency_missing(missing_dep: str):
    result = _run_import_with_blocked_imports([missing_dep])
    assert result.returncode == 0, (
        f"Base `import pdstools` should not require optional dependency `{missing_dep}`.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_import_pdstools_with_many_optional_dependencies_missing():
    result = _run_import_with_blocked_imports(OPTIONAL_IMPORTS)
    assert result.returncode == 0, (
        "Base `import pdstools` should not require optional dependencies.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

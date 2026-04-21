"""Tests verifying optional dependencies surface as MissingDependenciesException.

When an optional dependency is missing, pdstools should raise a
``MissingDependenciesException`` with a clear message (including the
dependency name and, when applicable, the extras group to install),
rather than a raw ``ImportError``.

See https://github.com/pegasystems/pega-datascientist-tools/issues/436.
"""

import asyncio
import builtins
import sys

import pytest

from pdstools.utils.namespaces import MissingDependenciesException


@pytest.fixture
def hide_module(monkeypatch):
    """Return a helper that makes ``import <name>`` raise ImportError."""

    def _hide(name: str) -> None:
        real_import = builtins.__import__

        def fake_import(modname, globals=None, locals=None, fromlist=(), level=0):
            if modname == name or modname.startswith(f"{name}."):
                raise ImportError(f"Simulated missing module: {name}")
            return real_import(modname, globals, locals, fromlist, level)

        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setattr(builtins, "__import__", fake_import)

    return _hide


def test_s3_missing_aioboto3_raises_missing_dependencies(hide_module):
    from pdstools.pega_io.S3 import S3Data

    hide_module("aioboto3")

    with pytest.raises(MissingDependenciesException) as excinfo:
        asyncio.run(S3Data(bucket_name="dummy-bucket").get_files("some/prefix"))

    assert "aioboto3" in str(excinfo.value)
    assert "pega_io" in str(excinfo.value)


def test_anonymization_missing_polars_hash_raises_missing_dependencies(hide_module, tmp_path):
    from pdstools.pega_io.Anonymization import Anonymization

    hide_module("polars_hash")

    anon = Anonymization(
        path_to_files=str(tmp_path / "*.json"),
        temporary_path=str(tmp_path / "tmp"),
    )

    with pytest.raises(MissingDependenciesException) as excinfo:
        anon.process(chunked_files=[])

    assert "polars-hash" in str(excinfo.value)
    assert "pega_io" in str(excinfo.value)

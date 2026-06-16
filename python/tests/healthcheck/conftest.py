"""Healthcheck test fixtures.

Isolates Quarto's per-user cache directories per pytest-xdist worker so that
parallel renders don't race on the shared Deno KV sass-cache sqlite file. The
underlying bug is in Quarto (the sass cache opens a Deno KV store at a fixed
per-user path that isn't safe for concurrent writers); symptom is an
intermittent ``ERROR: database is locked: Error code 5: The database file is
locked`` originating from ``sassCache`` during ``quarto render``.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_quarto_cache(tmp_path_factory, worker_id):
    cache_root = tmp_path_factory.mktemp(f"quarto-{worker_id}", numbered=False)
    previous = {key: os.environ.get(key) for key in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME")}
    os.environ["XDG_DATA_HOME"] = str(cache_root / "data")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "cache")
    os.environ["XDG_CONFIG_HOME"] = str(cache_root / "config")
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

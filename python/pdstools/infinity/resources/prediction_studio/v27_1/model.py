from __future__ import annotations

from ..v26_1.model import AsyncModel as AsyncModelV26_1
from ..v26_1.model import Model as ModelV26_1
from ..v26_1.model import _ModelEndpoints


class _Modelv27_1Mixin:
    """Use v5 model endpoints with the shared v26 model behavior."""

    _endpoints = _ModelEndpoints("v5", "v5")


class Model(_Modelv27_1Mixin, ModelV26_1):
    """v27 Model using v5 endpoints."""


class AsyncModel(_Modelv27_1Mixin, AsyncModelV26_1):
    """v27 async Model using v5 endpoints."""

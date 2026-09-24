from __future__ import annotations

from ...v26_1.prediction._mixin import _PredictionEndpoints, _Predictionv26_1Mixin


class _Predictionv27_1Mixin(_Predictionv26_1Mixin):
    """Use v5 endpoints with the shared v26 prediction behavior."""

    _endpoints = _PredictionEndpoints("v5", "v5", "v5", "v5", "v5")

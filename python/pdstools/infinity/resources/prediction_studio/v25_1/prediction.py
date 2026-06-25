from __future__ import annotations

from ..v26_1.prediction._async import AsyncPrediction as _AsyncPredictionv26_1
from ..v26_1.prediction._mixin import _Predictionv26_1Mixin
from ..v26_1.prediction._sync import Prediction as _Predictionv26_1


class _Predictionv25_1Mixin(_Predictionv26_1Mixin):
    """v25 Prediction — ``type`` field present, no ``performance`` fields.

    Overrides ``__init__`` to skip ``_Predictionv26_1Mixin.__init__`` (which
    sets ``self.performance`` / ``self.performance_measure``) and delegates
    directly to the base ``_PredictionMixin``.  All other v26 API methods
    are inherited unchanged.
    """

    def __init__(
        self,
        client,
        *,
        predictionId: str,
        label: str,
        type: str | None = None,
        status: str | None = None,
        lastUpdateTime: str | None = None,
        objective: str | None = None,
        subject: str | None = None,
        **kwargs,
    ):
        # Jump over _Predictionv26_1Mixin.__init__ to avoid setting
        # self.performance and self.performance_measure.
        super(_Predictionv26_1Mixin, self).__init__(
            client=client,
            predictionId=predictionId,
            label=label,
            status=status,
            lastUpdateTime=lastUpdateTime,
            objective=objective,
            subject=subject,
        )
        self.type = type


class Prediction(_Predictionv25_1Mixin, _Predictionv26_1):
    """v25 sync Prediction — inherits all v26 methods, no performance fields."""


class AsyncPrediction(_Predictionv25_1Mixin, _AsyncPredictionv26_1):
    """v25 async Prediction — inherits all v26 methods, no performance fields."""

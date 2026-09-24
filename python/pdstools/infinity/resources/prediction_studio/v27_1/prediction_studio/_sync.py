from __future__ import annotations

from .....internal._resource import api_method
from ...v26_1.prediction_studio import PredictionStudio as PredictionStudioV26_1
from ..datamart_export import DatamartExport
from ..model import Model
from ..prediction import Prediction
from ..repository import Repository
from ._mixin import _PredictionStudiov27_1Mixin


class PredictionStudio(_PredictionStudiov27_1Mixin, PredictionStudioV26_1):
    """v27 studio with shared operations and v5-specific repository parsing."""

    _model_cls = Model
    _prediction_cls = Prediction
    _datamart_export_cls = DatamartExport

    @api_method
    async def repository(self) -> Repository:
        """Get repository name from v5 settings; other details are unavailable."""
        general_settings = await self._a_general_settings()
        storage = general_settings.get("storage") or {}
        return Repository(
            client=self._client,
            repository_name=storage.get("value"),
            type=None,
            bucket_name=None,
            root_path=None,
            datamart_export_location=None,
        )

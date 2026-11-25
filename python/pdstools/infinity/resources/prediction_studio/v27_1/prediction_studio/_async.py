from __future__ import annotations

from ...v26_1.prediction_studio import AsyncPredictionStudio as AsyncPredictionStudioV26_1
from ..datamart_export import AsyncDatamartExport
from ..model import AsyncModel
from ..prediction import AsyncPrediction
from ..repository import AsyncRepository
from ._mixin import _PredictionStudiov27_1Mixin


class AsyncPredictionStudio(_PredictionStudiov27_1Mixin, AsyncPredictionStudioV26_1):
    """v27 async studio with shared operations and v5 repository parsing."""

    _model_cls = AsyncModel
    _prediction_cls = AsyncPrediction
    _datamart_export_cls = AsyncDatamartExport

    async def repository(self) -> AsyncRepository:
        """Get repository name from v5 settings; other details are unavailable."""
        general_settings = await self._a_general_settings()
        storage = general_settings.get("storage") or {}
        return AsyncRepository(
            client=self._client,
            repository_name=storage.get("value"),
            type=None,
            bucket_name=None,
            root_path=None,
            datamart_export_location=None,
        )

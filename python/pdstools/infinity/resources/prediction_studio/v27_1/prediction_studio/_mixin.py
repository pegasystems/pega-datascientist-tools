from __future__ import annotations

from .....internal._resource import api_method
from ...v26_1.prediction_studio._mixin import _PredictionStudiov26_1Mixin, _StudioEndpoints
from ..model_upload import UploadedModel


class _PredictionStudiov27_1Mixin(_PredictionStudiov26_1Mixin):
    """Share studio operations while reading v5 categories from settings."""

    version: str = "27.1"
    _endpoints = _StudioEndpoints("v5", "v5", "v5", "v5", "v5", "v5", "v5")
    _uploaded_model_cls = UploadedModel

    async def _a_general_settings(self) -> dict:
        """Fetch general settings from the v5 settings response."""
        response = await self._a_get(f"/prweb/api/PredictionStudio/{self._endpoints.settings}/settings")
        settings = response.get("settings") or {}
        return settings.get("generalSettings") or {}

    @api_method
    async def get_model_categories(self):
        """Get model categories from v5 general settings."""
        general_settings = await self._a_general_settings()
        return list(general_settings.get("modelCategories") or [])

from __future__ import annotations

from ..v24_2.datamart_export import AsyncDatamartExport as AsyncDatamartExportPrevious
from ..v24_2.datamart_export import DatamartExport as DatamartExportPrevious


class _DatamartExportv27_1Mixin:
    """v27 DatamartExport business logic — defined once.

    Add new or overridden methods here.
    """

    _EXPORT_STATUS_ENDPOINT = "/prweb/api/PredictionStudio/v5/datamart/export/{reference_id}"


class DatamartExport(_DatamartExportv27_1Mixin, DatamartExportPrevious):
    pass


class AsyncDatamartExport(_DatamartExportv27_1Mixin, AsyncDatamartExportPrevious):
    pass

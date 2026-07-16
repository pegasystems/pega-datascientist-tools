from __future__ import annotations

from ..v24_2.datamart_export import AsyncDatamartExport as AsyncDatamartExportPrevious
from ..v24_2.datamart_export import DatamartExport as DatamartExportPrevious


class _DatamartExportv27_1Mixin:
    """v27 DatamartExport business logic — defined once.

    Add new or overridden methods here.
    """


class DatamartExport(_DatamartExportv27_1Mixin, DatamartExportPrevious):
    pass


class AsyncDatamartExport(_DatamartExportv27_1Mixin, AsyncDatamartExportPrevious):
    pass

from __future__ import annotations

from ..v24_2.datamart_export import AsyncDatamartExport as AsyncDatamartExportPrevious
from ..v24_2.datamart_export import DatamartExport as DatamartExportPrevious


class _DatamartExportv26_1Mixin:
    """v26 DatamartExport business logic — defined once.

    Add new or overridden methods here.
    """


class DatamartExport(_DatamartExportv26_1Mixin, DatamartExportPrevious):
    pass


class AsyncDatamartExport(_DatamartExportv26_1Mixin, AsyncDatamartExportPrevious):
    pass

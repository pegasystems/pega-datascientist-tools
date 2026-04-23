from __future__ import annotations

# python/pdstools/decision_analyzer/data_read_utils.py
"""Decision Analyzer data-reading utilities.

The Action Analysis readers (``read_nested_zip_files``, ``read_gzipped_data``,
``read_gzipped_ndjson_directory``) have moved to :mod:`pdstools.pega_io` so all
user-facing file reads funnel through one inspectable surface. They are
re-exported here for back-compat with any external caller importing from the
old location; new code should import from ``pdstools.pega_io``.
"""

import polars as pl

from ..pega_io.action_analysis import (
    read_gzipped_data,
    read_gzipped_ndjson_directory,
    read_nested_zip_files,
)
from .utils import ColumnResolver
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .column_schema import TableConfig

__all__ = [
    "read_gzipped_data",
    "read_gzipped_ndjson_directory",
    "read_nested_zip_files",
    "validate_columns",
]


def validate_columns(
    df: pl.LazyFrame,
    extract_type: dict[str, TableConfig],
) -> tuple[bool, str | None]:
    """Validate that default columns from table definition exist in the dataframe.

    This function checks if required columns exist in the data, accounting for
    the fact that columns may be present under either their source name or
    their target label name.

    Parameters
    ----------
    df : pl.LazyFrame
        The dataframe to validate
    extract_type : dict[str, TableConfig]
        Table configuration mapping column names to their properties

    Returns
    -------
    tuple[bool, str | None]
        tuple containing validation success (bool) and error message (str or None)

    """
    resolver = ColumnResolver(
        table_definition=extract_type,
        raw_columns=set(df.collect_schema().names()),
    )
    missing_columns = resolver.get_missing_columns()

    if missing_columns:
        return (
            False,
            f"The following default columns are missing: {', '.join(missing_columns)}",
        )
    return True, None

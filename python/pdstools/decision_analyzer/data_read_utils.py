# python/pdstools/decision_analyzer/data_read_utils.py
import gzip
import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl

from ..pega_io.File import _is_artifact
from .column_schema import TableConfig
from .utils import ColumnResolver


# Decision Analyzer specific data reading utilities
# Core read_data is now in pega_io.File


def read_nested_zip_files(file_buffer) -> pl.DataFrame:
    """Read Pega Action Analysis export format (nested archive with gzipped NDJSON).

    Pega's Action Analysis feature exports decision data as a ZIP archive containing
    multiple inner files with `.zip` extensions. Despite the extension, these inner files
    are **gzipped NDJSON** (not ZIP archives). This function handles this format by:
    1. Opening the outer ZIP archive
    2. Treating each inner `.zip` file as gzipped NDJSON
    3. Decompressing and concatenating all data into a single DataFrame

    This format is used for high-volume decision event exports where data is partitioned
    across multiple compressed files.

    Parameters
    ----------
    file_buffer : UploadedFile or BytesIO
        ZIP archive buffer (e.g., from Streamlit file upload) containing inner
        gzipped NDJSON files with misleading `.zip` extensions.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame from all inner files, with consistent column ordering.

    Notes
    -----
    This is specific to Pega Action Analysis exports. Modern exports may use
    hive-partitioned parquet directories instead, which can be read with read_data().

    """
    dfs: list[pl.DataFrame] = []
    columns: list[str] = []

    with zipfile.ZipFile(file_buffer, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".zip") and not _is_artifact(Path(file_name).name):
                with zip_ref.open(file_name) as f:
                    data = BytesIO(f.read())
                    df = read_gzipped_data(data)
                    if df is not None and columns == []:
                        columns = df.columns
                    if df is not None:
                        dfs.append(df.select(columns))

    return pl.concat(dfs, rechunk=True)


def read_gzipped_data(data: BytesIO) -> pl.DataFrame | None:
    """Read a single gzipped NDJSON chunk from Pega Action Analysis export.

    Helper function for read_nested_zip_files(). Reads one inner file from the
    Action Analysis export format, decompresses the gzipped content, and parses
    the NDJSON data.

    Parameters
    ----------
    data : BytesIO
        Gzipped NDJSON data (from an inner file in Action Analysis export).

    Returns
    -------
    pl.DataFrame | None
        Polars DataFrame, or None if decompression/parsing fails.

    Notes
    -----
    Returns None on errors to allow processing remaining files even if some are corrupted.

    """
    try:
        with gzip.open(data, "rb") as file:
            file_content = file.read()
            return pl.read_ndjson(BytesIO(file_content)).lazy()  # type: ignore[return-value]
    except Exception as e:
        print(f"Error reading gzipped data: {e}")
        return None


def read_gzipped_ndjson_directory(path: str) -> pl.DataFrame:
    """Read directory of Pega Action Analysis gzipped NDJSON files.

    For extracted Action Analysis exports, this function recursively finds all files with
    `.zip` extension (which are actually gzipped NDJSON, not ZIP archives) and concatenates
    them into a single DataFrame. Useful when the outer archive has been extracted to disk.

    Parameters
    ----------
    path : str
        Path to directory containing gzipped NDJSON files with `.zip` extension
        (from extracted Action Analysis export).

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame from all files with consistent column ordering.

    Notes
    -----
    This is specific to Pega Action Analysis exports. For normal data reading
    (including hive-partitioned directories), use read_data() from pega_io instead.

    """
    dfs: list[pl.DataFrame] = []
    columns: list[str] = []

    for file_path in sorted(Path(path).rglob("*.zip")):
        if any(part.startswith(".") or _is_artifact(part) for part in file_path.parts):
            continue
        with gzip.open(file_path, "rb") as file:
            file_content = file.read()
            df = pl.read_ndjson(BytesIO(file_content)).lazy()
            if columns == []:
                columns = df.columns
            dfs.append(df.select(columns))  # type: ignore[arg-type]

    return pl.concat(dfs, rechunk=True)


# Note: read_data and helper functions have been moved to pega_io.File for reuse
# Import them from there: from ..pega_io.File import read_data


def validate_columns(
    df: pl.LazyFrame,
    extract_type: dict[str, TableConfig],
) -> tuple[bool, str | None]:
    """Validate that default columns from table definition exist in the dataframe.

    This function checks if required columns exist in the data, accounting for
    the fact that columns may be present under either their source name or
    their target label name.

    Args:
        df: The dataframe to validate
        extract_type: Table configuration mapping column names to their properties

    Returns:
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

"""Readers for Pega Action Analysis export format.

Pega's Action Analysis feature exports decision data as a ZIP archive containing
multiple inner files with `.zip` extensions. Despite the extension, these inner
files are **gzipped NDJSON** (not ZIP archives). The functions in this module
handle that format — both in-memory (``BytesIO`` from Streamlit uploads) and on
disk (extracted export directories).

Path-taking readers in this module are the single funnel for Action Analysis
disk reads; CodeQL ``py/path-injection`` is scoped out here at the config level.
"""

import gzip
import logging
import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl

from .File import _is_artifact, _scan_by_extension

logger = logging.getLogger(__name__)


def read_nested_zip_files(file_buffer) -> pl.LazyFrame:
    """Read Pega Action Analysis export format (nested archive with gzipped NDJSON).

    Pega's Action Analysis feature exports decision data as a ZIP archive containing
    multiple inner files with `.zip` extensions. Despite the extension, these inner files
    are **gzipped NDJSON** (not ZIP archives). This function handles this format by:
    1. Opening the outer ZIP archive
    2. Treating each inner `.zip` file as gzipped NDJSON
    3. Decompressing and concatenating all data into a single LazyFrame

    This format is used for high-volume decision event exports where data is partitioned
    across multiple compressed files.

    Parameters
    ----------
    file_buffer : UploadedFile or BytesIO
        ZIP archive buffer (e.g., from Streamlit file upload) containing inner
        gzipped NDJSON files with misleading `.zip` extensions.

    Returns
    -------
    pl.LazyFrame
        Concatenated LazyFrame from all inner files, with consistent column ordering.
        Call ``.collect()`` to materialize.

    Notes
    -----
    This is specific to Pega Action Analysis exports. Modern exports may use
    hive-partitioned parquet directories instead, which can be read with read_data().

    """
    dfs: list[pl.LazyFrame] = []
    columns: list[str] = []

    with zipfile.ZipFile(file_buffer, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".zip") and not _is_artifact(Path(file_name).name):
                with zip_ref.open(file_name) as f:
                    data = BytesIO(f.read())
                    df = read_gzipped_data(data)
                    if df is not None and columns == []:
                        columns = df.collect_schema().names()
                    if df is not None:
                        dfs.append(df.select(columns))

    return pl.concat(dfs, rechunk=True)


def read_gzipped_data(data: BytesIO) -> pl.LazyFrame | None:
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
    pl.LazyFrame | None
        Polars LazyFrame, or None if decompression/parsing fails.

    Notes
    -----
    Returns None on errors to allow processing remaining files even if some are corrupted.

    """
    try:
        with gzip.open(data, "rb") as file:
            file_content = file.read()
            return _scan_by_extension(BytesIO(file_content), ".json")
    except Exception as e:
        logger.warning("Error reading gzipped data: %s", e)
        return None


def read_gzipped_ndjson_directory(path: str) -> pl.LazyFrame:
    """Read directory of Pega Action Analysis gzipped NDJSON files.

    For extracted Action Analysis exports, this function recursively finds all files with
    `.zip` extension (which are actually gzipped NDJSON, not ZIP archives) and concatenates
    them into a single LazyFrame. Useful when the outer archive has been extracted to disk.

    Parameters
    ----------
    path : str
        Path to directory containing gzipped NDJSON files with `.zip` extension
        (from extracted Action Analysis export).

    Returns
    -------
    pl.LazyFrame
        Concatenated LazyFrame from all files with consistent column ordering.
        Call ``.collect()`` to materialize.

    Notes
    -----
    This is specific to Pega Action Analysis exports. For normal data reading
    (including hive-partitioned directories), use read_data() from pega_io instead.

    """
    dfs: list[pl.LazyFrame] = []
    columns: list[str] = []

    for file_path in sorted(Path(path).rglob("*.zip")):
        if any(part.startswith(".") or _is_artifact(part) for part in file_path.parts):
            continue
        with gzip.open(file_path, "rb") as file:
            file_content = file.read()
            df = _scan_by_extension(BytesIO(file_content), ".json")
            if columns == []:
                columns = df.collect_schema().names()
            dfs.append(df.select(columns))

    return pl.concat(dfs, rechunk=True)

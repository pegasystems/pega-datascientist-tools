# python/pdstools/decision_analyzer/data_read_utils.py
import gzip
import os
import shutil
import tarfile
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl

from .column_schema import TableConfig
from .utils import ColumnResolver

# Extensions that read_data knows how to handle.
_SUPPORTED_EXTENSIONS: set[str] = {
    ".parquet",
    ".csv",
    ".arrow",
    ".ndjson",
    ".jsonl",
    ".json",
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
}


def _is_artifact(name: str) -> bool:
    """Return True for OS-generated junk entries (macOS, Windows, etc.)."""
    return name.startswith("__MACOSX") or name.startswith("._") or name in {".DS_Store", "Thumbs.db", "desktop.ini"}


def _clean_artifacts(directory: str) -> None:
    """Remove OS-generated artifact files and directories after archive extraction.

    Polars glob patterns (e.g. ``**/*.parquet``) cannot skip hidden files or
    ``__MACOSX`` resource-fork directories, so we delete them from the
    extracted tree before scanning.
    """
    root = Path(directory)
    # Remove __MACOSX directories first (contains bulk of macOS resource forks)
    for macosx_dir in root.rglob("__MACOSX"):
        if macosx_dir.is_dir():
            shutil.rmtree(macosx_dir, ignore_errors=True)
    # Remove individual artifact files (._*, .DS_Store, Thumbs.db, etc.)
    for p in list(root.rglob(".*")):
        if p.is_file() and _is_artifact(p.name):
            p.unlink(missing_ok=True)
    for name in ("Thumbs.db", "desktop.ini"):
        for p in root.rglob(name):
            if p.is_file():
                p.unlink(missing_ok=True)


def read_nested_zip_files(file_buffer) -> pl.DataFrame:
    """Reads a zip file buffer (uploaded from Streamlit) that contains .zip files,
    which are in fact gzipped ndjson files. Extracts, reads, and concatenates
    them into a single Polars DataFrame.

    Parameters
    ----------
    file_buffer : UploadedFile
        The uploaded zip file buffer from Streamlit.

    Returns
    -------
    pl.DataFrame
        A concatenated Polars DataFrame containing the data from all gzipped ndjson files.

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
    """Reads gzipped ndjson data from a BytesIO object and returns a Polars DataFrame.

    Parameters
    ----------
    data : BytesIO
        The gzipped ndjson data.

    Returns
    -------
    pl.DataFrame | None
        The Polars DataFrame containing the data, or None if reading fails.

    """
    try:
        with gzip.open(data, "rb") as file:
            file_content = file.read()
            return pl.read_ndjson(BytesIO(file_content)).lazy()  # type: ignore[return-value]
    except Exception as e:
        print(f"Error reading gzipped data: {e}")
        return None


def read_gzips_with_zip_extension(path: str) -> pl.DataFrame:
    """Recursively finds all files with a .zip extension under the given directory,
    treats them as gzipped ndjson files, reads, and concatenates them into a single
    Polars DataFrame. Supports arbitrary directory depth.

    Parameters
    ----------
    path : str
        The path to the directory containing the .zip files.

    Returns
    -------
    pl.DataFrame
        A concatenated Polars DataFrame containing the data from all gzipped ndjson files.

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


def _is_tar_path(path: Path) -> bool:
    """Check if a path refers to a tar archive (plain or compressed)."""
    name = path.name.lower()
    return name.endswith((".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz"))


def _extract_tar(archive_path: Path) -> str:
    """Extract a tar archive to a temporary directory and return the path."""
    tmp_dir = tempfile.mkdtemp(prefix="pdstools_tar_")
    with tarfile.open(archive_path, mode="r:*") as tf:
        tf.extractall(tmp_dir, filter="data")
    return tmp_dir


def read_data(path):
    original_path = Path(path)  # save the original path
    extension = None  # Initialize extension to None
    if original_path.is_dir():
        # It's a directory, possibly hive-partitioned at arbitrary depth.
        # Find the first supported data file, skipping hidden/OS-artifact entries.
        for dirpath, dirs, files in os.walk(str(original_path)):
            dirs[:] = [d for d in dirs if not d.startswith(".") and not _is_artifact(d)]
            for file in sorted(files):
                if file.startswith(".") or _is_artifact(file):
                    continue
                ext = Path(file).suffix
                if ext in _SUPPORTED_EXTENSIONS:
                    extension = ext
                    break
            if extension:
                break
        # Use a recursive glob — works regardless of partition depth.
        path = original_path / f"**/*{extension}"
    else:
        # It's a file, so we read based on the extension
        extension = original_path.suffix

    # Handle tar archives first (covers .tar, .tar.gz, .tgz, etc.)
    if not original_path.is_dir() and _is_tar_path(original_path):
        tmp_dir = _extract_tar(original_path)
        _clean_artifacts(tmp_dir)
        return read_data(tmp_dir)

    if extension == ".parquet":
        df = pl.scan_parquet(path)
    elif extension == ".csv":
        df = pl.scan_csv(path)
    elif extension == ".arrow":
        df = pl.scan_ipc(path)
    elif extension in {".ndjson", ".jsonl", ".json"}:
        df = pl.scan_ndjson(path)
    elif extension == ".zip":
        if original_path.is_file():
            # Single zip file: extract to temp dir and read the contents
            tmp_dir = tempfile.mkdtemp(prefix="pdstools_zip_")
            with zipfile.ZipFile(original_path, "r") as zf:
                zf.extractall(tmp_dir)
            _clean_artifacts(tmp_dir)
            df = read_data(tmp_dir)
        else:
            # Directory of .zip files (legacy gzipped ndjson)
            df = read_gzips_with_zip_extension(str(original_path))
    elif extension is None:
        raise ValueError("No data files found in directory")
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return df


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
